# Provider Migration: Rust → Swift (mlx-swift-lm), CLI-only

## Status (as of v0.5.0 cut)

**Phases 0–5: complete, including Phase 4b (true continuous batching).**
Phase 6 (cutover): integration test fixtures landed; coordinator-side
`mlx-swift` acceptance was already in place; `LatestProviderVersion` is
bumped to 0.5.0; `install.sh` is the pure-Swift bundle installer (Python
and vllm-mlx fully removed). The SwiftPM metallib build-tool plugin is
the only remaining deferred item.

Continuous batching is **the production inference path** (no parallel
implementations). `ProviderCore.BatchScheduler` is now an actor that wraps
a single shared `BatchGenerator` (ported from `mlx_lm.generate`); all
concurrent requests are merged into one batched forward pass per step.
Validated bit-identical against single-stream greedy reference on:
- Qwen3 0.6B-8bit (dense), B=2 and B=4-ragged
- Qwen3.5 0.8B-MLX-4bit (hybrid SSM + attention), B=2
- Gemma 4 26B-A4B-it-8bit (MoE, 26 GB), B=2

Plus eviction-and-admission live test: row 0 finishes mid-batch, row C is
admitted into its slot, row C's tokens match a solo run, and row B
(running through the eviction) also matches its solo run. This validates
`BatchKVCache.filterBatched` + `extendBatched` end-to-end.

Per-row samplers (`makeRowSampler`) thread `temperature` / `top_p` /
`top_k` / `seed` from `ChatCompletionRequest` straight into the
`BatchGenerator`'s per-row sampler slot, so each concurrent request gets
its own sampling configuration.

CLI surface ships these subcommands: `serve`, `start`, `stop`, `status`,
`doctor`, `models {list,catalog,download,remove}`, `enroll`, `unenroll`,
`login`, `logout`, `logs`, `autoupdate`, `benchmark`, `update`. The
foreground `start --foreground` path also handles PID-file single-instance
enforcement, `caffeinate`-based sleep prevention, panic-hook telemetry,
and the metallib hash that's surfaced under `template_hashes["mlx_metallib"]`
in registration + attestation responses.

## Overview

Migrate the Rust provider to Swift, replacing vllm-mlx with mlx-swift-lm for
native inference. Eliminates Python entirely. **CLI-only**: the legacy SwiftUI
menu bar app at `app/EigenInference/` and the Swift FFI bridge at `enclave/`
have been deleted; the migration ships exactly two binaries — `darkbloom`
(provider CLI) and `eigeninference-enclave` (Secure Enclave helper).

### Architecture: Before vs After

```
BEFORE (3 languages, 2 processes, 1 GUI):
┌─────────────────────┐     ┌──────────────────────────────┐
│ Swift Menu Bar App   │────>│ Rust Provider Binary         │
│ (thin GUI, 6K lines) │     │ ├── WebSocket → Coordinator  │
└─────────────────────┘     │ ├── PyO3 Python sandbox      │
                             │ │   └── vllm-mlx engine      │
                             │ │       └── MLX → Metal → GPU │
                             │ ├── OR: HTTP proxy → subprocess│
                             │ │   └── vllm-mlx serve (child)│
                             │ ├── Security hardening       │
                             │ ├── FFI → Enclave Swift lib  │
                             │ └── Telemetry, config, etc.  │
                             └──────────────────────────────┘

AFTER (1 language, 1 process, no GUI):
┌─────────────────────────────────────────┐
│ darkbloom (Swift CLI)                    │
│ ├── WebSocket → Coordinator              │
│ ├── mlx-swift-lm (direct library call)   │
│ │   └── MLX → Metal → GPU               │
│ ├── (Phase 4) BatchScheduler             │
│ ├── Security hardening                   │
│ ├── Secure Enclave (native, no FFI)      │
│ ├── Sodium (X25519/XSalsa20Poly1305)     │
│ └── Telemetry, config, models, etc.      │
└─────────────────────────────────────────┘
                  +
┌─────────────────────────────────────────┐
│ eigeninference-enclave (Swift CLI)       │
│ Stateless attestation/sign helper used   │
│ by install.sh during device provisioning.│
└─────────────────────────────────────────┘
```

The menu bar GUI is out of scope for this migration. Operators interact with
the provider through `darkbloom` directly (`serve`, `start`, `stop`, `status`,
`doctor`, `models`, `login`, `logout`, `benchmark`, `update`, `verify`).

### Key Numbers

| Metric | Value |
|---|---|
| Code eliminated | ~7,100 lines (Python sandbox, subprocess mgmt, FFI bridges, HTTP proxy) |
| Code ported | ~12,300 Rust lines → ~8,000 Swift lines |
| Net new code | ~3,500 lines (batch scheduler + OpenAI formatter + tokenizer adapter + metallib build glue) |
| Final owned codebase | ~12K Swift (down from ~49K Rust+Swift+Python) |
| Inference library (free) | 56K lines (mlx-swift-lm, forked under Gajesh2007) |
| Estimated timeline | 6-8 weeks |

### Monorepo Structure

Forked dependencies live as git submodules in `libs/`:

```
d-inference/
├── libs/
│   ├── mlx-swift/          # github.com/Layr-Labs/mlx-swift (submodule)
│   └── mlx-swift-lm/       # github.com/Layr-Labs/mlx-swift-lm (submodule)
├── provider-swift/          # Swift provider package — CLI only
│   ├── Package.swift        # SPM manifest (references ../libs/ as local deps)
│   └── Sources/
│       ├── ProviderCore/                  # shared library
│       ├── darkbloom/                     # provider CLI (executable)
│       └── eigeninference-enclave-cli/    # SE attestation helper (executable)
├── provider/                # Rust provider (retired at cutover)
├── coordinator/             # Go coordinator (unchanged)
├── console-ui/              # Next.js frontend (unchanged)
└── ...
```

The legacy `app/EigenInference/` SwiftUI menu bar app and the legacy
`enclave/` Swift FFI bridge have been deleted — both were either irrelevant to
a CLI-only migration (the app) or have been re-implemented natively in
`ProviderCore` and `eigeninference-enclave-cli` (the enclave).

---

## Inventory: What Goes Where

### Eliminated (no port needed, deleted from repo)

| Path | Lines | Why |
|---|---|---|
| `provider/src/inference.rs` | 1,251 | mlx-swift-lm replaces PyO3 engine |
| `provider/src/proxy.rs` | 1,710 | No HTTP proxy — inference is in-process |
| `provider/src/backend/mod.rs` | 1,373 | No subprocess to manage |
| `provider/src/backend/vllm_mlx.rs` | 322 | No vllm-mlx subprocess |
| `provider/src/secure_enclave_key.rs` | 703 | Replaced by ProviderCore + CryptoKit |
| Python runtime hashing in `security.rs` | ~500 | No Python |
| `enclave/` (entire directory) | 478 | Reimplemented natively in ProviderCore + eigeninference-enclave-cli |
| `app/EigenInference/` (entire directory) | 6,037 | Out of scope: CLI-only migration |
| `provider/src/wallet.rs` | 258 | Legacy, drop |
| `scripts/build-bundle.sh`, `scripts/bundle-app.sh`, `scripts/sign-hardened.sh`, `scripts/build-bridge-app.sh` | ~750 | App/Python/Rust bundle scripts; replaced by `release-swift.yml` |
| `.github/workflows/release.yml` | 575 | Legacy Rust+Python+app pipeline; replaced by `release-swift.yml` |
| **Total** | **~13,957** | |

### Already in ProviderCore (Phase 0–3 complete)

These exist in `provider-swift/Sources/ProviderCore/` today and replace the
named Rust files. They were ported from the now-deleted `app/` and `enclave/`
trees plus written from scratch where needed.

| Module | Replaces Rust | Status |
|---|---|---|
| `Security/SecureEnclaveIdentity.swift` | `secure_enclave_key.rs` | ✓ |
| `Security/AttestationBuilder.swift` | (attestation portion) | ✓ |
| `Config/ProviderConfig.swift` | `config.rs` | ✓ |
| `Security/SecurityHardening.swift`, `SecurityFoundation.swift`, `AntiDebug.swift`, `EnvironmentScrubber.swift`, `BinaryHasher.swift` | `security.rs` (non-Python parts) | ✓ |
| `Models/ModelScanner.swift`, `WeightHasher.swift` | `models.rs` | ✓ |
| `Service/LaunchAgent.swift` | `service.rs` | ✓ |
| `Hardware/HardwareDetector.swift`, `SystemMetrics.swift` | `hardware.rs` | ✓ |
| `Crypto/NodeKeyPair.swift` | `crypto.rs` | ✓ (libsodium NaCl box) |
| `Coordinator/CoordinatorClient.swift` | `coordinator.rs` | ✓ |
| `Protocol/Messages.swift`, `ProtocolCodec.swift`, `Types.swift`, `Enums.swift` | `protocol.rs` | ✓ |
| `Inference/InferenceEngine.swift` (+ supporting types) | `inference.rs` | ✓ |
| `Server/StandaloneServer.swift` | `server.rs` | ✓ |
| `Scheduling/Schedule.swift` | `scheduling.rs` | ✓ |
| `ProviderLoop.swift` | top-level driver in `main.rs` | ✓ |
| `Auth/DeviceAuth.swift` | (login flow in `main.rs`) | ✓ |
| `Update/SelfUpdater.swift` | (update flow in `main.rs`) | ✓ |
| `Benchmark/ModelBenchmark.swift` | (benchmark cmd in `main.rs`) | ✓ |

### Must be ported (Rust → new Swift)

| Rust File | Lines | Swift Est. | Notes |
|---|---|---|---|
| `main.rs` (state machine + CLI) | 7,611 | ~3,000 | Split into ProviderCore + CLI |
| `coordinator.rs` | 1,527 | ~800 | URLSessionWebSocketTask or SwiftNIO |
| `protocol.rs` | 1,255 | ~600 | Mechanical Codable translation |
| `security.rs` (non-Python parts) | ~943 | ~500 | ptrace, SIP, binary hash — same syscalls |
| `hardware.rs` | 670 | ~400 | sysctl + system_profiler |
| `crypto.rs` | 462 | ~200 | swift-sodium (XSalsa20Poly1305 + Curve25519) |
| `config.rs` | 461 | ~150 | Extend existing ConfigManager |
| `scheduling.rs` | 439 | ~300 | Pure logic, no platform deps |
| `server.rs` | 641 | ~300 | Hummingbird for standalone mode |
| `service.rs` | 210 | ~80 | Extend existing LaunchAgentManager |
| `telemetry/` | 980 | ~300 | Extend existing TelemetryReporter |
| `models.rs` | 1,082 | ~350 | Extend existing ModelManager |

### Net new (doesn't exist in current codebase)

| Component | Est. Lines | Why |
|---|---|---|
| `BatchScheduler` (continuous) | ~3,000 | mlx-swift-lm is single-stream; see Phase 4 |
| OpenAI response formatter | ~400 | Format mlx-swift-lm output as SSE chunks |
| Hummingbird HTTP server | ~300 | Standalone mode |
| Tokenizer adapter | ~50 | Bridge `swift-transformers` `Tokenizer` ↔ `MLXLMCommon.Tokenizer` |
| Metallib build glue | ~150 | SwiftPM doesn't ship a metallib for `Cmlx`; need plugin or vendored .metallib |
| Speculative decoding wiring | 0 | First-class in `mlx-swift-lm` — config only |

---

## Protocol Surface

11 message types, 11 sub-structs, 3 enums. JSON with `snake_case` keys and `"type"` discriminator.

### Provider → Coordinator (7 types)

1. **`register`** — Hardware, models, attestation, auth token, privacy capabilities, runtime hashes. **Critical**: `attestation` field must preserve raw JSON bytes.
2. **`heartbeat`** — Status (idle/serving), active model, warm models, stats, system metrics, backend capacity.
3. **`inference_accepted`** — Ack request, extends coordinator wait window.
4. **`inference_response_chunk`** — SSE chunk (plaintext `data` or `encrypted_data` NaCl box).
5. **`inference_complete`** — Usage info, optional SE signature of response hash.
6. **`inference_error`** — Error with status code.
7. **`attestation_response`** — Nonce signature, SE public key, security status fields, model/runtime hashes.

### Coordinator → Provider (4 types)

1. **`inference_request`** — Request ID + encrypted body (NaCl box).
2. **`cancel`** — Cancel in-flight request by ID.
3. **`attestation_challenge`** — Nonce + timestamp for SE signing.
4. **`runtime_status`** — Hash verification result with mismatches.

### Key sub-structs

`HardwareInfo`, `CpuCores`, `ModelInfo`, `PrivacyCapabilities` (9 bool flags), `ProviderStats`, `SystemMetrics`, `BackendCapacity`, `BackendSlotCapacity`, `UsageInfo`, `EncryptedPayload`, `RuntimeMismatch`.

### Serialization constraints

- Optional fields use `omitempty` / `skip_serializing_if`
- `attestation` must be raw bytes (`JSONSerialization`, not `Codable` decode+re-encode)
- `HardwareInfo` type mismatches between Go/Rust (u64 vs float64 on wire) — Swift should use `Double`
- `ModelInfo` has Rust-only fields (`parameters`, `estimated_memory_gb`) that Go ignores

---

## Phase Plan

### Phase 0: Foundation (Week 1)

**Goal**: mlx-swift-lm compiles, basic inference works from Swift, all build/runtime prereqs are documented and reproducible.

**Build / runtime prerequisites** (ALL must land before Phase 1):

- [ ] **Metallib build path.** `swift build` against vendored `libs/mlx-swift` does **not** produce a metallib — SwiftPM does not auto-compile the `.metal` files in `Source/Cmlx/mlx-generated/metal/`, and the kernels are explicitly excluded from the `Cmlx` target's C++ build. Pick exactly one of:
  - (preferred for app target) Build through Xcode / `xcodebuild`; Xcode's SwiftPM integration compiles + bundles `.metal` files automatically.
  - (preferred for CLI / CI) Add a SwiftPM **build-tool plugin** to `libs/mlx-swift` that runs `xcrun -sdk macosx metal -O3 -ffast-math -c …` over `Source/Cmlx/mlx-generated/metal/` and `xcrun -sdk macosx metallib …` to emit `mlx.metallib`, then declares a resource on the `Cmlx` target. ~150 lines, one-time.
  - (cheapest, version-pinned) Vendor a prebuilt `mlx.metallib` from the matching `mlx==0.31.x` Python wheel under `libs/mlx-swift/Source/Cmlx/Resources/` and reference it via `.copy("mlx.metallib")`. Pins kernel set to that wheel.
  Without one of these, `swift build -c release` produces a binary that throws `Failed to load the default metallib` on first MLX call. CI MUST fail loudly if no metallib was produced.
- [ ] **Tokenizer integration**. Decide on one of:
  - Depend on `huggingface/swift-transformers` (>= 1.3.0) and use `Tokenizers.AutoTokenizer.from(modelFolder:)` directly (no hub client required). Hand-roll a ~50-line `MLXLMCommon.TokenizerLoader` that wraps it. Recommended — matches current Rust behavior of "load tokenizer from local cache directory."
  - Pull in `MLXHuggingFace` macros (transitively brings `swift-huggingface`, BoringSSL, NIO, Jinja, yyjson, Crypto, Collections — non-trivial closure). Avoid unless we want the hub downloader.
- [ ] **Chat-template fidelity check.** For every model in `coordinator/internal/registry/catalog.go`, render `tokenizer.applyChatTemplate(messages:tools:additionalContext:)` against `swift-transformers` and compare token-for-token to a Python `tokenizers` baseline. Jinja parity is the silent breakage point; cheap to verify once.
- [ ] Create `provider-swift/Package.swift` referencing `../libs/mlx-swift-lm` as local dependency.
- [ ] Verify `~/.cache/huggingface/hub/` models load without download (using free `loadModelContainer(from: directory, using:)`).
- [ ] Benchmark tok/s vs vllm-mlx on the same model on the same hardware (Qwen3-8B-4bit, Gemma 4 26B-A4B-8bit). MLX C++ kernels are identical between mlx-swift and Python mlx-lm, so ±10 % is expected.

**mlx-swift-lm API for local loading**:

```swift
// Free function tries VLM trampoline first, then LLM, via ModelFactoryRegistry.
let container = try await loadModelContainer(
    from: modelDirectory, using: tokenizerLoader
)

// High level
let session = ChatSession(container, generateParameters: params)
for try await chunk in session.streamResponse(to: prompt) { ... }

// Low level (lets you cancel, pull GenerateCompletionInfo, etc.)
await container.perform { context in
    let lmInput = try await context.processor.prepare(input: userInput)
    let stream = try generate(input: lmInput, parameters: params, context: context)
    for await event in stream {
        switch event {
        case .chunk(let s): ...
        case .toolCall(let call): ...
        case .info(let info): ...        // promptTokensPerSecond, tokensPerSecond, stopReason
        }
    }
}
```

**Model factory selection**: many catalog models (Gemma 3 12B/27B, Gemma 4 26B-A4B, Gemma 4 31B, Qwen 3.5 27B/35B-A3B) ship as `…ForConditionalGeneration` and only resolve through `VLMModelFactory`. Several `model_type` strings (e.g. `gemma4`) are registered in **both** `LLMModelFactory` and `VLMModelFactory`. Use the free `loadModelContainer(from:using:)` (which iterates `[VLM, LLM]` via `ModelFactoryRegistry`) or do explicit "VLM first, fall back to LLM."

**Risk gate**: If tok/s is >10 % worse than vllm-mlx, investigate before proceeding. If the metallib step is unsolved, do not proceed.

---

### Phase 1: Protocol + WebSocket (Week 2)

**Goal**: Swift provider connects to coordinator, registers, heartbeats.

- [ ] Port `protocol.rs` → Swift Codable structs (~600 lines)
  - All 11 message types with `CodingKeys` for snake_case
  - Outer enum with `"type"` discriminator handled by manual `init(from:)` / `encode(to:)` (Swift Codable can't match Rust's `#[serde(tag = "type", rename_all = "snake_case")]` automatically without boilerplate)
  - `attestation` as raw `Data` slot — round-trip the JSON bytes verbatim, never decode + re-encode
  - Enums: `ProviderStatus`, `ChipFamily`, `ChipTier`, `ThermalState`
  - Round-trip tests against fixture JSON captured from the running Rust provider
- [ ] Port `hardware.rs` → Swift (~400 lines)
  - `sysctlbyname` for memory, CPU cores
  - `system_profiler SPDisplaysDataType -json` for GPU cores
  - Chip family/tier parsing, bandwidth lookup table
  - Live metrics: `vm_stat`-style pressure, `vm.loadavg`-derived CPU usage, `pmset -g therm`-derived thermal state
- [ ] Build WebSocket coordinator client (~800 lines)
  - `URLSessionWebSocketTask` + reconnect loop + exponential backoff (cap 30 s)
  - Registration on connect (must include attestation as raw JSON bytes)
  - Heartbeat timer with shared state (5 s default)
  - Ping/pong keepalive with 30 s timeout
  - Message dispatch: inference request, cancel, attestation challenge, runtime status

**Test**: Point at dev coordinator (`api.dev.darkbloom.xyz`), provider appears in dashboard.

---

### Phase 2: Single-Request Inference (Week 3)

**Goal**: End-to-end inference through the full pipeline.

- [ ] Inference engine wrapper around mlx-swift-lm
  - Load model from HuggingFace cache directory (re-uses Phase 0 verifier)
  - OpenAI chat messages → `UserInput` → `LMInput`
  - Stream `AsyncStream<Generation>` → format as SSE `data:` lines
  - Track prompt/completion tokens from `GenerateCompletionInfo`
- [ ] Port `crypto.rs` → swift-sodium (~200 lines)
  - **Critical**: NaCl `crypto_box` uses XSalsa20-Poly1305 + Curve25519 (HSalsa20 derivation). Apple `CryptoKit` ships ChaCha20-Poly1305 only; it is NOT wire-compatible with the Rust `crypto_box` crate or Go's `golang.org/x/crypto/nacl/box`.
  - Use `jedisct1/swift-sodium` (libsodium SwiftPM wrapper) or vendor libsodium directly. `Curve25519.KeyAgreement.PrivateKey` from CryptoKit is fine for the X25519 keypair, but the AEAD must be libsodium's `crypto_box_easy` / `crypto_box_open_easy`.
  - Round-trip test: encrypt with libsodium-Swift, decrypt with `crypto_box`-Rust (fixture). Same in reverse. Add to `tests/test_crypto_interop.py` parity suite.
  - Wire format: `nonce(24 bytes) || ciphertext`, base64-encoded, alongside `ephemeral_public_key` base64.
- [ ] Wire inference into coordinator dispatch
  - `inference_request` → decrypt → generate → encrypt chunks → send
  - `inference_accepted` / `inference_response_chunk` / `inference_complete`
  - `cancel` → cancel Swift `Task`. Note: cancellation latency floor is ~one decode step (~10–20 ms) because mlx-swift-lm only checks `Task.isCancelled` between iterator steps. Same as today's Rust provider.
- [ ] Port `models.rs` → extend ModelManager (~350 lines)
  - Scan HuggingFace cache for MLX models (uses `HardwareInfo.memory_available_gb` filter)
  - Read `config.json` metadata (`model_type`, parameters)
  - On-demand SHA-256 weight hashing via `compute_weight_hash(model_id)`
  - `resolve_local_path(model_id)` → snapshot dir for backend loading
  - Catalog filter — only models in the coordinator's catalog are reported

**Test**: Chat request through console-ui → coordinator → Swift provider → response streams back. Cross-language NaCl box test passes.

---

### Phase 3: Security Hardening (Week 4) — parallel with Phase 4

**Goal**: Match Rust provider's security posture.

- [ ] Port security primitives (~500 lines)
  - `PT_DENY_ATTACH`: `ptrace(PT_DENY_ATTACH, 0, nil, 0)` via `Darwin`
  - SIP check: `csr_get_active_config()` C symbol if present, else parse `csrutil status`
  - Binary self-hash via SHA-256 of `Bundle.main.executableURL`
  - Core dump disable (`setrlimit(RLIMIT_CORE, 0)`), environment scrubbing
- [x] Secure Enclave is native in `provider-swift`
  - Implemented in `provider-swift/Sources/ProviderCore/Security/SecureEnclaveIdentity.swift` and `AttestationBuilder.swift`. The legacy `enclave/` directory and its FFI bridge were deleted.
  - `eigeninference-enclave-cli` exposes `attest`, `sign`, `info`, `wallet-address` for `install.sh`-style use.
- [ ] Allow RDMA-enabled posture without Hypervisor
  - RDMA status is reported and signed in challenge responses
  - Safety policy is based on RDMA-aware runtime registration discipline, not Stage 2 page tables
  - The current Rust provider's `hypervisor.rs` (Stage 2 page tables) is dropped — see Risks
- [ ] Update `PrivacyCapabilities` for Swift-native provider
  - `text_backend_inprocess` = true (always — mlx-swift-lm is in-process by definition)
  - `text_proxy_disabled` = true (no HTTP proxy)
  - `python_runtime_locked`, `dangerous_modules_blocked` = drop (deprecated, see Phase 6)
  - Keep `sip_enabled`, `anti_debug_enabled`, `core_dumps_disabled`, `env_scrubbed`
  - `hypervisor_active` always false on the Swift provider (RDMA discipline replaces it)

**Test**: Coordinator shows `trust: hardware`. All doctor checks pass.

---

### Phase 4: Continuous Batching — **COMPLETE (v0.5.0)**

**Goal**: Serve multiple concurrent requests efficiently with bit-identical
greedy correctness vs single-stream.

**Implementation lives in the Swift fork of mlx-swift-lm** (modifications
opted into when we vendored the library) and the production scheduler in
`provider-swift`. Architecturally this is the same design as upstream
Python `mlx_lm` (`BatchKVCache` + `BatchGenerator` from PR #443) ported
to Swift, plus a hybrid-cache extension that supports both full-attention
and SSM-style layers in the same model.

**Shipped components**

- `BatchKVCache` (`libs/mlx-swift-lm/Libraries/MLXLMCommon/BatchKVCache.swift`)
  — left-padded right-justified storage, per-row offsets, `update`,
  `filter`, `extend`, `extract`, `merge` primitives.
- `BatchedCache` protocol — abstraction over batched-KV and batched-SSM
  caches; both `BatchKVCache` and `ArraysCache` (parent of `MambaCache`)
  conform.
- `SequenceStateMachine` — per-row trie-based stop-sequence detector,
  ported from `mlx_lm.generate`.
- `PromptProcessingBatch` — chunked prefill phase; outputs a
  `GenerationBatch`.
- `GenerationBatch` — decode phase; per-row sampling + stop detection;
  one forward pass per step emits per-row tokens.
- `BatchGenerator` — orchestrator: `insert(prompts:)` queues requests,
  `next()` admits + decodes one step, finished rows are filtered and
  their slots become available for new admissions on the next call.
  `makeBatchedCache` probes `model.newCache()` and allocates the right
  batched cache type per layer (full-attention → `BatchKVCache`,
  SSM-style → `MambaCache` / `ArraysCache`), so hybrid models like
  Qwen 3.5 work with the same engine.
- `makeRowSampler(temperature:topP:topK:seed:)` — per-row sampler
  factory; greedy at `temperature == 0`, otherwise scaled-logits +
  optional top-K + optional top-P + optional seeded `MLXRandom.categorical`.
  Each row keeps its own PRNG key (split forward each step).

**Production scheduler** (`provider-swift/Sources/ProviderCore/Inference/BatchScheduler.swift`)

- One actor wraps a single shared `BatchGenerator`.
- Detached worker task drives `gen.next()` in a tight loop, sleeping
  5 ms when there's no work; calls into the actor only for short
  critical sections (state updates + response dispatch). Avoids the
  classic actor-deadlock pattern where `cancel`/`submit` sit behind a
  long-running worker.
- `submit(request:)` tokenizes via `applyChatTemplate`, builds a
  per-row sampler from the request's `temperature` / `top_p` / `top_k`
  / `seed`, and inserts into the engine.
- `cancel(requestId:)` finishes the request's stream with an error;
  the BatchGenerator naturally drops the row on its next step.
- `unloadModel()` cancels the worker, closes the generator, drops
  references — releasing the model is the unload (no `unload()` API
  on `ModelContainer`).

**Validation**

| Test | Model | Status |
|------|-------|--------|
| Bit-identical greedy diff vs solo | Qwen3 0.6B-8bit B=2 | ✓ |
| Bit-identical greedy diff vs solo | Qwen3 0.6B-8bit B=4 ragged | ✓ |
| Bit-identical greedy diff vs solo | Qwen3.5 0.8B-MLX-4bit (hybrid SSM+attention) B=2 | ✓ |
| Bit-identical greedy diff vs solo | Gemma 4 26B-A4B-it-8bit (MoE, 26 GB) B=2 | ✓ |
| Same-prompt determinism across positions | Qwen3 0.6B B=4 | ✓ |
| Eviction + re-admission deterministic match | Qwen3 0.6B (A finishes, C admitted, B running) | ✓ |
| Sampler unit tests (greedy, top-K, top-P, seed) | logits-only | ✓ (7 tests) |

Drift on close-call argmax after the first ~half of the window is
expected (vLLM, mlx-lm, sglang behave the same way under bf16/fp16
reduction-order changes). Tests assert bit-identity on the prefix and
log the divergence point on the rest.

**Idle timeout**: drop the `ModelContainer` after 1 hour of inactivity;
lazy-reload by calling `loadContainer(from:using:)` again. There is no
`unload()` method — releasing the reference is unload. Bracket every
generation with `WiredMemoryTicket.withWiredLimit` so concurrent
loads/unloads don't fight over wired memory.

**Pending follow-ups (non-blocking)**

- Per-row repetition / presence / frequency penalty processors. Today
  the wire fields exist on `ChatCompletionRequest` but are pass-through
  only.
- Throughput benchmark suite (B=1/2/4 vs single-stream tok/s) — useful
  for tracking regressions but not a correctness gate.

---

### Phase 5: CLI surface polish (Week 7)

**Goal**: `darkbloom` CLI is feature-complete and ergonomic. No GUI work in
this migration — see "Out of scope" below.

- [ ] CLI subcommands feature-complete
  - `darkbloom serve` (foreground, used by launchd)
  - `darkbloom start` / `stop` (launchd install + control)
  - `darkbloom status` / `doctor` / `models`
  - `darkbloom login` / `logout` (RFC 8628 device-code flow)
  - `darkbloom benchmark` (Phase 0 verifier; tok/s vs vllm-mlx golden numbers)
  - `darkbloom update` (self-update via signed bundle)
  - `darkbloom verify` (Phase 0 fidelity: tokenizer chat-template parity, model load smoke test)
- [ ] Port remaining utilities into ProviderCore
  - Telemetry wire types (keep in sync with `coordinator/internal/protocol/telemetry.go` and `console-ui/src/lib/telemetry-types.ts`)
  - `darkbloom doctor` performs the same checks as the legacy app's diagnostics view: SIP, Hardened Runtime, anti-debug, Secure Enclave availability, available memory headroom, model catalog presence
- [ ] Standalone HTTP server polish (Hummingbird)
  - `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
  - Match the existing OpenAI-compatible shape exposed by `provider/src/server.rs`

**Out of scope (intentional)**:

- No SwiftUI menu bar app, no `.app` bundle, no DMG.
- No in-process integration with the legacy `app/EigenInference/` (it has been
  deleted from the repo). End users interact with the provider via the CLI
  directly. If a GUI is needed later it can ship as a separate package that
  shells out to `darkbloom` exactly the way the legacy app shelled out to the
  Rust binary.

---

### Phase 6: Feature Parity + Cutover (Week 8)

**Goal**: Full test suite passes, deploy to production, retire Rust.

- [ ] Integration tests
  - Protocol round-trip (Swift↔Go) — fixture-based, captured from running Rust provider
  - Mock coordinator end-to-end
  - Multi-model serving, E2E encryption, cancellation, reconnection
  - NaCl box cross-language interop (Swift ↔ Rust ↔ Go)
- [ ] Build infrastructure
  - CI: `swift build -c release` + code signing + notarization
  - New workflow: `.github/workflows/release-swift.yml` (no Rust, no Python)
  - `release-swift.yml` produces the bundle entirely; `scripts/build-bundle.sh` / `scripts/bundle-app.sh` / `scripts/sign-hardened.sh` / `scripts/build-bridge-app.sh` were deleted alongside the legacy app/enclave dirs.
  - Update `scripts/install.sh` (and the embedded copy at `coordinator/internal/api/install.sh`) to install the Swift bundle: drop the Python-runtime download, drop the vllm-mlx zip, drop the libpython rpath rewrite, drop the Mach-O sign-tree loop. New shape: download tarball → verify SHA-256 + codesign → extract `bin/{darkbloom,eigeninference-enclave,mlx.metallib}` to `~/.darkbloom/bin/`.
- [ ] Coordinator compatibility
  - New `backend` value: `"mlx-swift"` vs `"vllm-mlx"`
  - Deprecate `python_hash` / `runtime_hash` (drop from registration)
  - Add `mlx_swift_lm_version` / `mlx_swift_version` to registration
  - Coordinator should accept providers without `python_hash` for `backend == "mlx-swift"`
- [ ] Production validation
  - Deploy to testbench machines, side-by-side comparison with Rust provider
  - 48-hour soak: stability, reconnection, thermal behavior, memory leaks
  - Gradual fleet rollout: 5 % → 25 % → 100 %

---

## Critical Path

```
Phase 0 ──> Phase 1 ──> Phase 2 ──┬──> Phase 3 (security) ──> Phase 5 ──> Phase 6
                                   └──> Phase 4 (batching, deferrable) ──┘
```

Phases 3 and 4 are independent and can run in parallel. **Phase 4b shipped in v0.5.0** — continuous batching is the production inference path.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Metallib not built by SwiftPM | Binary fails on first MLX call | Phase 0 build-tool plugin or vendored .metallib resource. CI must fail loudly if no metallib produced. |
| NaCl box wire compat (XSalsa20Poly1305 vs ChaCha20Poly1305) | E2E encryption breaks silently | swift-sodium dependency, NOT CryptoKit. Cross-language fixture tests in `tests/test_crypto_interop.py` before Phase 2. |
| tok/s regression | Slower inference | Phase 0 benchmark gate. Same MLX C++ backend. |
| WebSocket stability | Provider drops offline | URLSessionWebSocketTask + reconnect testing. Fallback: SwiftNIO. |
| Continuous batching bugs | Wrong output / silent corruption | Static-first (4a), continuous-deferrable (4b). Extensive attention mask tests. Per-model RoPE call-site audit. |
| Raw JSON attestation | Signature breaks | `JSONSerialization` for byte preservation. Don't decode + re-encode. Round-trip test. |
| Architecture gap | Model X unsupported | Phase 0: render and tokenize chat for every catalog model on swift-transformers and verify token-for-token parity with Python. |
| Swift 6 concurrency | Sendable / actor errors | mlx-swift-lm targets Swift 6.1. Follow their patterns. |
| Tokenizer Jinja drift | Wrong prompts in production | Phase 0 chat-template fidelity check. |
| Hypervisor.framework removed | Lower trust score | Acceptable — RDMA discipline replaces Stage 2 page tables. Coordinator `MIN_TRUST` policy unchanged. |

## Dependencies

| Package | Source | Purpose |
|---|---|---|
| mlx-swift | `libs/mlx-swift` (submodule, forked) | MLX array operations, Metal GPU compute |
| mlx-swift-lm | `libs/mlx-swift-lm` (submodule, forked) | Inference engine (50+ architectures) |
| swift-transformers | `huggingface/swift-transformers` >= 1.3.0 | Tokenizer + Jinja chat template |
| swift-sodium | `jedisct1/swift-sodium` | NaCl `crypto_box` (XSalsa20Poly1305 + Curve25519) — wire-compatible with Rust `crypto_box` and Go `nacl/box` |
| swift-argument-parser | `apple/swift-argument-parser` | CLI subcommand parsing |
| hummingbird | `hummingbird-project/hummingbird` | Standalone HTTP server |
| CryptoKit | system | SHA-256, HKDF, P-256 (Secure Enclave) |
| Security.framework | system | Secure Enclave P-256 keys |

**Min deployment target: macOS 14 (Sonoma)** — matches `libs/mlx-swift-lm` and `libs/mlx-swift` declared platforms. The current Rust provider already supports macOS 14, so this preserves the install base. Bumping to 15 is only required if we want a 15-only API (Hypervisor.framework Stage 2 was the only candidate and we're removing it in Phase 3).

## Free wins from mlx-swift-lm

These are first-class in the upstream library and require no porting beyond a config flag or a registration value:

- **Speculative decoding** via `SpeculativeTokenIterator` and `generate(... draftModel: ..., draftCache: ..., numDraftTokens: ...)`. For paired draft+target with shared tokenizer (e.g. Qwen3-0.6B drafting Qwen3-8B), this is a sizeable throughput win for zero new code.
- **KV cache quantization** via `GenerateParameters.kvBits` / `kvGroupSize` / `quantizedKVStart`. Trades a small quality drop for ~50 % less GPU memory at long contexts.
- **Wired memory policy** via `WiredMemoryTicket.withWiredLimit` — admission control that we can hook into `BackendCapacity` reporting.
- **RotatingKVCache** for sliding-window models (Gemma sliding attention, etc.) — already used automatically when `maxKVSize` is set.
- **Tool-call parsing** via `ToolCallProcessor` — handles `.json`, `.glm4`, `.lfm2`, `.mistral`, `.xmlFunction` formats out of the box.
