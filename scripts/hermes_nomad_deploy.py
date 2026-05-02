#!/usr/bin/env python3
"""Deterministic GitHub webhook -> Nomad HITL deployment helper.

This script intentionally keeps the LLM out of the deploy decision path. A
Hermes webhook/agent run can invoke `propose` and deliver the emitted approval
text, but only `approve` executes `nomad job run` and it revalidates the frozen
repo/ref/SHA before doing so.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = "deploy/hermes/nomad-hitl.example.json"
SUCCESS_PLAN_EXIT_CODES = {0, 2}


class DeployError(RuntimeError):
    pass


@dataclass
class CommandResult:
    argv: list[str]
    cwd: Path | None
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expand_path(value: str, base: Path | None = None) -> Path:
    value = os.path.expandvars(os.path.expanduser(value))
    p = Path(value)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


def load_config(path: str | Path) -> dict[str, Any]:
    p = expand_path(str(path), repo_root())
    with p.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("approval_ttl_seconds", 1800)
    cfg.setdefault("deploy_path_prefixes", ["nomad/jobs/", "deploy/nomad/"])
    cfg.setdefault("job_globs", ["nomad/jobs/*.nomad", "nomad/jobs/**/*.nomad", "deploy/nomad/*.nomad", "deploy/nomad/**/*.nomad"])
    cfg.setdefault("require_signed_commit", False)
    cfg.setdefault("allowed_actors", [])
    cfg.setdefault("nomad_addr", "http://127.0.0.1:4646")
    cfg.setdefault("state_dir", "~/.hermes/deployments/d-inference/state")
    cfg.setdefault("work_dir", "~/.hermes/deployments/d-inference/work")
    return cfg


def run(argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = False) -> CommandResult:
    proc = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = CommandResult(argv=argv, cwd=cwd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    if check and proc.returncode != 0:
        raise DeployError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{result.combined}")
    return result


def read_text(path: str | None) -> str | None:
    if not path:
        return None
    return expand_path(path).read_text(encoding="utf-8").strip()


def verify_github_signature(raw_body: bytes, signature_header: str, secret: str) -> None:
    if not signature_header.startswith("sha256="):
        raise DeployError("missing or invalid X-Hub-Signature-256 header")
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise DeployError("webhook signature mismatch")


def load_payload(path: str) -> tuple[dict[str, Any], bytes]:
    if path == "-":
        raw = sys.stdin.buffer.read()
    else:
        p = expand_path(path)
        raw = p.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def changed_files_from_push(payload: dict[str, Any]) -> list[str]:
    files: set[str] = set()
    for commit in payload.get("commits") or []:
        for key in ("added", "modified", "removed"):
            for item in commit.get(key) or []:
                files.add(item)
    head = payload.get("head_commit") or {}
    for key in ("added", "modified", "removed"):
        for item in head.get(key) or []:
            files.add(item)
    return sorted(files)


def removed_files_from_push(payload: dict[str, Any]) -> list[str]:
    files: set[str] = set()
    for commit in payload.get("commits") or []:
        for item in commit.get("removed") or []:
            files.add(item)
    head = payload.get("head_commit") or {}
    for item in head.get("removed") or []:
        files.add(item)
    return sorted(files)


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def under_any_prefix(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def selected_job_files(changed_files: list[str], cfg: dict[str, Any]) -> list[str]:
    return sorted(f for f in changed_files if matches_any(f, cfg["job_globs"]))


def risky_changes(payload: dict[str, Any], changed_files: list[str], job_files: list[str], plan_outputs: dict[str, str]) -> list[str]:
    risks: list[str] = []
    removed = removed_files_from_push(payload)
    removed_jobs = [f for f in removed if f in job_files or f.endswith(".nomad")]
    if removed_jobs:
        risks.append("Nomad job file removed: " + ", ".join(removed_jobs))
    runtime_files = [f for f in changed_files if f.endswith((".sh", ".zsh", ".bash")) or "/bin/" in f]
    if runtime_files:
        risks.append("Runtime wrapper/script changed: " + ", ".join(runtime_files))
    secretish = [f for f in changed_files if any(token in f.lower() for token in ("secret", ".env", "token", "key"))]
    if secretish:
        risks.append("Secret-like path changed: " + ", ".join(secretish))
    scary_terms = ("destroy", "destructive", "delete", "remove", "stop", "purge", "migrate")
    for job, output in plan_outputs.items():
        low = output.lower()
        hits = sorted({term for term in scary_terms if term in low})
        if hits:
            risks.append(f"Plan for {job} contains risk terms: {', '.join(hits)}")
    return risks


def ensure_checkout(cfg: dict[str, Any], sha: str) -> Path:
    work_dir = expand_path(cfg["work_dir"])
    checkout = work_dir / "repo"
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_url = cfg["repo_url"]
    ref = cfg["ref"]

    if not (checkout / ".git").exists():
        run(["git", "clone", "--no-checkout", repo_url, str(checkout)], check=True)
    run(["git", "remote", "set-url", "origin", repo_url], cwd=checkout, check=True)
    remote_ref = "refs/remotes/origin/" + ref.removeprefix("refs/heads/")
    run(["git", "fetch", "--prune", "origin", f"+{ref}:{remote_ref}"], cwd=checkout, check=True)

    # Verify the requested SHA exists on the configured ref before checkout.
    branch_ref = remote_ref
    run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=checkout, check=True)
    ancestry = run(["git", "merge-base", "--is-ancestor", sha, branch_ref], cwd=checkout)
    if ancestry.returncode != 0:
        raise DeployError(f"commit {sha} is not reachable from configured ref {ref}")
    run(["git", "checkout", "--force", sha], cwd=checkout, check=True)
    run(["git", "clean", "-fdx"], cwd=checkout, check=True)
    return checkout


def commit_signature(checkout: Path, sha: str) -> dict[str, str]:
    result = run(["git", "log", "-1", "--format=%G?|%GK|%GS|%s", sha], cwd=checkout, check=True)
    status, key, signer, subject = (result.stdout.strip().split("|", 3) + ["", "", "", ""])[:4]
    return {"status": status, "key": key, "signer": signer, "subject": subject}


def nomad_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["NOMAD_ADDR"] = cfg["nomad_addr"]
    token = read_text(cfg.get("nomad_token_file"))
    if token:
        env["NOMAD_TOKEN"] = token
    return env


def nomad_validate_and_plan(checkout: Path, cfg: dict[str, Any], job_files: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    env = nomad_env(cfg)
    validates: dict[str, str] = {}
    plans: dict[str, str] = {}
    if not job_files:
        return validates, plans
    if not shutil.which("nomad"):
        raise DeployError("nomad CLI not found on PATH")
    for rel in job_files:
        job = checkout / rel
        if not job.exists():
            raise DeployError(f"selected job file does not exist at checked out SHA: {rel}")
        validate = run(["nomad", "job", "validate", str(job)], cwd=checkout, env=env)
        validates[rel] = validate.combined
        if validate.returncode != 0:
            raise DeployError(f"nomad job validate failed for {rel}\n{validate.combined}")
        plan = run(["nomad", "job", "plan", str(job)], cwd=checkout, env=env)
        plans[rel] = plan.combined
        if plan.returncode not in SUCCESS_PLAN_EXIT_CODES:
            raise DeployError(f"nomad job plan failed for {rel} with exit {plan.returncode}\n{plan.combined}")
    return validates, plans


def plan_hash(plans: dict[str, str]) -> str:
    material = json.dumps(plans, sort_keys=True).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def state_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    state = expand_path(cfg["state_dir"])
    pending = state / "pending"
    closed = state / "closed"
    pending.mkdir(parents=True, exist_ok=True)
    closed.mkdir(parents=True, exist_ok=True)
    return state, pending, closed


def write_pending(cfg: dict[str, Any], proposal: dict[str, Any]) -> Path:
    _, pending, _ = state_dirs(cfg)
    path = pending / f"{proposal['id']}.json"
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def load_pending(cfg: dict[str, Any], deploy_id: str) -> tuple[Path, dict[str, Any]]:
    _, pending, _ = state_dirs(cfg)
    path = pending / f"{deploy_id}.json"
    if not path.exists():
        raise DeployError(f"pending deployment not found: {deploy_id}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def close_pending(cfg: dict[str, Any], path: Path, proposal: dict[str, Any], status: str) -> Path:
    _, _, closed = state_dirs(cfg)
    proposal["status"] = status
    proposal["closed_at"] = int(time.time())
    dest = closed / path.name
    dest.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(dest, 0o600)
    path.unlink(missing_ok=True)
    return dest


def approval_message(proposal: dict[str, Any]) -> str:
    risks = proposal.get("risks") or []
    risk_text = "none" if not risks else "\n" + "\n".join(f"- {r}" for r in risks)
    jobs = proposal.get("job_files") or []
    jobs_text = "none" if not jobs else "\n" + "\n".join(f"- {j}" for j in jobs)
    changed = proposal.get("changed_files") or []
    changed_preview = changed[:12]
    changed_text = "none" if not changed_preview else "\n" + "\n".join(f"- {c}" for c in changed_preview)
    if len(changed) > len(changed_preview):
        changed_text += f"\n- … {len(changed) - len(changed_preview)} more"
    destructive_suffix = " --destructive-ok" if risks else ""
    expires = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(proposal["expires_at"]))
    return textwrap.dedent(
        f"""
        Nomad deploy approval requested

        ID: {proposal['id']}
        Repo: {proposal['repo']}
        Ref: {proposal['ref']}
        Commit: {proposal['sha'][:12]}
        Actor: {proposal.get('actor') or 'unknown'}
        Signature: {proposal.get('commit_signature', {}).get('status', '?')} {proposal.get('commit_signature', {}).get('key', '')}

        Jobs:{jobs_text}

        Changed files:{changed_text}

        Risk flags: {risk_text}

        Approve: approve {proposal['id']}{destructive_suffix}
        Deny: deny {proposal['id']}
        Expires: {expires}
        """
    ).strip()


def propose(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    payload, raw_body = load_payload(args.payload_file)
    if args.signature and args.secret_file:
        verify_github_signature(raw_body, args.signature, read_text(args.secret_file) or "")

    if args.event != "push":
        raise DeployError(f"ignored unsupported GitHub event: {args.event}")
    repo = (payload.get("repository") or {}).get("full_name")
    ref = payload.get("ref")
    sha = payload.get("after")
    actor = ((payload.get("sender") or {}).get("login") or (payload.get("pusher") or {}).get("name") or "")

    if repo != cfg["repo"]:
        raise DeployError(f"repo not allowed: {repo}")
    if ref != cfg["ref"]:
        raise DeployError(f"ref not allowed: {ref}")
    if not sha or set(sha) == {"0"}:
        raise DeployError("push payload has empty/deleted after SHA")
    allowed_actors = cfg.get("allowed_actors") or []
    if allowed_actors and actor not in allowed_actors:
        raise DeployError(f"actor not allowed: {actor}")

    changed = changed_files_from_push(payload)
    relevant = [f for f in changed if under_any_prefix(f, cfg["deploy_path_prefixes"])]
    if not relevant:
        print(json.dumps({"status": "ignored", "reason": "no deploy-relevant paths changed", "changed_files": changed}, indent=2))
        return 0

    jobs = selected_job_files(relevant, cfg)
    if not jobs:
        print(json.dumps({
            "status": "ignored",
            "reason": "deploy-support files changed but no Nomad job file changed; pair wrapper/config edits with a job edit or deploy manually",
            "relevant_files": relevant,
        }, indent=2))
        return 0

    checkout = ensure_checkout(cfg, sha)
    sig = commit_signature(checkout, sha)
    if cfg.get("require_signed_commit") and sig["status"] != "G":
        raise DeployError(f"commit signature is required but status is {sig['status']}")
    validates, plans = nomad_validate_and_plan(checkout, cfg, jobs)

    now = int(time.time())
    deploy_id = "deploy_" + sha[:12] + "_" + secrets.token_hex(3)
    proposal = {
        "id": deploy_id,
        "status": "pending",
        "created_at": now,
        "expires_at": now + int(cfg["approval_ttl_seconds"]),
        "github_delivery": args.delivery_id,
        "repo": repo,
        "ref": ref,
        "sha": sha,
        "actor": actor,
        "before": payload.get("before"),
        "changed_files": changed,
        "relevant_files": relevant,
        "job_files": jobs,
        "commit_signature": sig,
        "validate_outputs": validates,
        "plan_outputs": plans,
        "plan_hash": plan_hash(plans),
    }
    proposal["risks"] = risky_changes(payload, relevant, jobs, plans)
    path = write_pending(cfg, proposal)
    print(approval_message(proposal))
    print(f"\nPending file: {path}")
    return 0


def approve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path, proposal = load_pending(cfg, args.deploy_id)
    now = int(time.time())
    if now > int(proposal["expires_at"]):
        close_pending(cfg, path, proposal, "expired")
        raise DeployError(f"deployment expired: {args.deploy_id}")
    risks = proposal.get("risks") or []
    if risks and not args.destructive_ok:
        raise DeployError("risk flags present; re-run with --destructive-ok after reviewing the plan")

    checkout = ensure_checkout(cfg, proposal["sha"])
    sig = commit_signature(checkout, proposal["sha"])
    if cfg.get("require_signed_commit") and sig["status"] != "G":
        raise DeployError(f"commit signature is required but status is {sig['status']}")
    _, plans = nomad_validate_and_plan(checkout, cfg, proposal["job_files"])
    new_hash = plan_hash(plans)
    if new_hash != proposal.get("plan_hash") and not args.plan_changed_ok:
        raise DeployError("plan output changed since proposal; inspect and re-run with --plan-changed-ok if acceptable")

    env = nomad_env(cfg)
    run_outputs: dict[str, str] = {}
    for rel in proposal["job_files"]:
        job = checkout / rel
        result = run(["nomad", "job", "run", "-detach", str(job)], cwd=checkout, env=env)
        run_outputs[rel] = result.combined
        if result.returncode != 0:
            proposal["run_outputs"] = run_outputs
            close_pending(cfg, path, proposal, "failed")
            raise DeployError(f"nomad job run failed for {rel}\n{result.combined}")
    proposal["approved_at"] = now
    proposal["approved_by"] = args.approved_by or os.getenv("USER", "unknown")
    proposal["run_outputs"] = run_outputs
    dest = close_pending(cfg, path, proposal, "deployed")
    print(f"Deployed {args.deploy_id}; closed record: {dest}")
    for rel, output in run_outputs.items():
        print(f"\n## {rel}\n{output}")
    return 0


def deny(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path, proposal = load_pending(cfg, args.deploy_id)
    proposal["denied_by"] = args.denied_by or os.getenv("USER", "unknown")
    proposal["deny_reason"] = args.reason or ""
    dest = close_pending(cfg, path, proposal, "denied")
    print(f"Denied {args.deploy_id}; closed record: {dest}")
    return 0


def list_pending(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _, pending, _ = state_dirs(cfg)
    rows = []
    for p in sorted(pending.glob("deploy_*.json")):
        item = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: item.get(k) for k in ("id", "repo", "ref", "sha", "actor", "expires_at", "job_files", "risks")})
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def show(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path, proposal = load_pending(cfg, args.deploy_id)
    if args.message:
        print(approval_message(proposal))
    else:
        print(path.read_text(encoding="utf-8"))
    return 0


def self_test(args: argparse.Namespace) -> int:
    """Run offline safety smoke tests that do not need Nomad or network."""
    cfg = load_config(args.config)
    sha = run(["git", "rev-parse", "HEAD"], cwd=repo_root(), check=True).stdout.strip()
    def run_payload(payload: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as fh:
            json.dump(payload, fh)
            payload_path = fh.name
        try:
            ns = argparse.Namespace(
                config=args.config,
                payload_file=payload_path,
                event="push",
                delivery_id="self-test",
                signature="",
                secret_file="",
            )
            rc = propose(ns)
            if rc != 0:
                raise DeployError(f"expected ignored proposal to exit 0, got {rc}")
        finally:
            Path(payload_path).unlink(missing_ok=True)

    run_payload({
        "ref": cfg["ref"],
        "before": "0" * 40,
        "after": sha,
        "repository": {"full_name": cfg["repo"]},
        "sender": {"login": "self-test"},
        "commits": [{"added": [], "modified": ["README.md"], "removed": []}],
        "head_commit": {"added": [], "modified": ["README.md"], "removed": []},
    })
    run_payload({
        "ref": cfg["ref"],
        "before": "0" * 40,
        "after": sha,
        "repository": {"full_name": cfg["repo"]},
        "sender": {"login": "self-test"},
        "commits": [{"added": [], "modified": [".hermes/deployments/bin/run-service.sh"], "removed": []}],
        "head_commit": {"added": [], "modified": [".hermes/deployments/bin/run-service.sh"], "removed": []},
    })

    secret = "test-secret"
    raw = b'{"ok":true}'
    sig = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    verify_github_signature(raw, sig, secret)
    try:
        verify_github_signature(raw, sig, "wrong")
    except DeployError:
        pass
    else:
        raise DeployError("signature verifier accepted the wrong secret")
    print("self-test passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="JSON config path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propose", help="verify a GitHub push payload, run validate/plan, create pending approval")
    p.add_argument("--payload-file", required=True, help="raw GitHub JSON payload path, or '-' for stdin")
    p.add_argument("--event", default="push")
    p.add_argument("--delivery-id", default="")
    p.add_argument("--signature", default="", help="X-Hub-Signature-256 value for raw payload verification")
    p.add_argument("--secret-file", default="", help="file containing GitHub webhook HMAC secret")
    p.set_defaults(func=propose)

    p = sub.add_parser("approve", help="approve and execute a pending deployment")
    p.add_argument("deploy_id")
    p.add_argument("--approved-by", default="")
    p.add_argument("--destructive-ok", action="store_true")
    p.add_argument("--plan-changed-ok", action="store_true")
    p.set_defaults(func=approve)

    p = sub.add_parser("deny", help="deny a pending deployment")
    p.add_argument("deploy_id")
    p.add_argument("--denied-by", default="")
    p.add_argument("--reason", default="")
    p.set_defaults(func=deny)

    p = sub.add_parser("list", help="list pending deployments")
    p.set_defaults(func=list_pending)

    p = sub.add_parser("show", help="show a pending deployment record")
    p.add_argument("deploy_id")
    p.add_argument("--message", action="store_true")
    p.set_defaults(func=show)

    p = sub.add_parser("self-test", help="run offline safety smoke tests")
    p.set_defaults(func=self_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DeployError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
