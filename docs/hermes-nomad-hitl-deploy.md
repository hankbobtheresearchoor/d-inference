# Hermes HITL Nomad Deploys

This runbook describes the safe deployment path for GitHub-triggered Nomad jobs.
GitHub can request a deployment, but it cannot deploy directly. Hermes verifies
and plans locally, then waits for human-in-the-loop approval before running
Nomad.

## Security model

```text
GitHub push
  -> GitHub webhook with X-Hub-Signature-256
  -> public Hermes webhook route
  -> deterministic local deploy helper verifies/filters/plans
  -> Telegram approval request
  -> human approve/deny
  -> local Nomad job run
```

GitHub stores only the webhook HMAC secret. It must not store:

- Nomad ACL tokens
- SSH keys into cluster machines
- Cloudflare tunnel credentials
- LAN-only service URLs or device credentials

The Hermes/Nomad host stores the matching webhook secret, Nomad token, and any
read-only Git credential needed to fetch the repository. Those values stay local
and should live under `~/.hermes/secrets/` or another host secret store, never in
this repository or chat logs.

## Files

- `scripts/hermes_nomad_deploy.py` — deterministic helper used by Hermes.
- `deploy/hermes/nomad-hitl.example.json` — non-secret config template.

Create the real config outside the repo, for example:

```bash
mkdir -p ~/.hermes/deployments/d-inference ~/.hermes/secrets
cp deploy/hermes/nomad-hitl.example.json ~/.hermes/deployments/d-inference/nomad-hitl.json
chmod 700 ~/.hermes/secrets ~/.hermes/deployments/d-inference
chmod 600 ~/.hermes/deployments/d-inference/nomad-hitl.json
```

Then write secrets locally:

```bash
# Do not paste values into chat/logs.
install -m 600 /dev/null ~/.hermes/secrets/github-d-inference-webhook.secret
install -m 600 /dev/null ~/.hermes/secrets/nomad.token
$EDITOR ~/.hermes/secrets/github-d-inference-webhook.secret
$EDITOR ~/.hermes/secrets/nomad.token
```

## GitHub webhook subscription

Create a Hermes webhook route that receives only GitHub `push` events. The exact
command depends on the public Hermes webhook URL, but the route should call the
helper in `propose` mode and deliver its stdout to Telegram.

Example prompt for a Hermes webhook subscription:

```text
A GitHub push webhook was received for Layr-Labs/d-inference.
Run this deterministic command; do not deploy unless a later human approval
command is received:

python3 /path/to/d-inference/scripts/hermes_nomad_deploy.py \
  --config ~/.hermes/deployments/d-inference/nomad-hitl.json \
  propose \
  --payload-file /path/to/raw-github-payload.json \
  --event push \
  --delivery-id <x-github-delivery> \
  --signature <x-hub-signature-256> \
  --secret-file ~/.hermes/secrets/github-d-inference-webhook.secret
```

Hermes webhook subscriptions vary by adapter version in how they expose raw
payloads and headers. If the prompt cannot pass a raw payload file and signature
header directly, add a thin local wrapper route that writes the exact request body
to a chmod-600 temp file and invokes the same `propose` command. The security
invariant is unchanged: signature verification and deploy gating happen inside
the deterministic helper.

GitHub webhook settings:

- Payload URL: Hermes webhook URL
- Content type: `application/json`
- Secret: value from `~/.hermes/secrets/github-d-inference-webhook.secret`
- Events: Just the `push` event
- SSL verification: enabled

## Proposal behavior

`propose` is intentionally read/plan only:

1. Verify `X-Hub-Signature-256` if a signature and secret file are provided.
2. Require GitHub event `push`.
3. Require repo `Layr-Labs/d-inference`.
4. Require ref `refs/heads/master`.
5. Optionally require the actor allowlist.
6. Ignore pushes that do not touch configured deploy paths.
7. Ignore deploy-support-only pushes unless a Nomad job file changed too.
8. Fetch the exact pushed SHA locally.
9. Verify the SHA is reachable from the configured ref.
10. Optionally require a good GPG signature.
11. Run `nomad job validate` and `nomad job plan` for changed Nomad job files.
12. Save a pending deployment under `~/.hermes/deployments/d-inference/state/pending/`.
13. Emit a Telegram-ready approval message.

Example local dry run with a saved GitHub payload:

```bash
python3 scripts/hermes_nomad_deploy.py \
  --config ~/.hermes/deployments/d-inference/nomad-hitl.json \
  propose \
  --payload-file /tmp/github-push.json \
  --event push \
  --delivery-id manual-test
```

## Approval flow

A human approves only after reading the plan summary:

```bash
python3 scripts/hermes_nomad_deploy.py \
  --config ~/.hermes/deployments/d-inference/nomad-hitl.json \
  approve deploy_<id> \
  --approved-by ethan
```

If the proposal has risk flags, approval must be explicit:

```bash
python3 scripts/hermes_nomad_deploy.py \
  --config ~/.hermes/deployments/d-inference/nomad-hitl.json \
  approve deploy_<id> \
  --approved-by ethan \
  --destructive-ok
```

`approve` re-fetches and re-checks the exact SHA, re-runs `nomad validate` and
`nomad plan`, compares the new plan hash to the proposal, then runs:

```bash
nomad job run -detach <job-file>
```

If the plan changed, it fails closed unless `--plan-changed-ok` is provided
after manual inspection.

Deny instead:

```bash
python3 scripts/hermes_nomad_deploy.py \
  --config ~/.hermes/deployments/d-inference/nomad-hitl.json \
  deny deploy_<id> \
  --denied-by ethan \
  --reason "needs review"
```

## Risk flags

The helper highlights proposals requiring extra attention:

- Nomad job file removal
- runtime wrapper/script changes
- secret-like path changes (`secret`, `.env`, `token`, `key`)
- risky words in `nomad plan` output such as `destroy`, `delete`, `remove`,
  `stop`, `purge`, or `migrate`

Risk flags do not auto-deploy. They require `--destructive-ok` on approval. Deploy-support-only changes without a paired Nomad job edit are ignored so they cannot create a misleading no-op approval.

## Operational notes

- Prefer a read-only deploy key or GitHub App credential on the Hermes host for
  local `git fetch`; do not put broad GitHub PATs in Actions secrets.
- Keep `require_signed_commit: true` for production.
- Keep the Nomad ACL token scoped to job deployment, not management, where
  possible.
- Store pending/closed records chmod 600; they may contain plan output and repo
  metadata.
- Hermes/LLM is allowed to report, summarize, and route approval messages, but
  the helper owns deploy authorization and execution checks.
