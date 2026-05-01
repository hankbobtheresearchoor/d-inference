# Continuous Batching — Status

End-to-end continuous batching is implemented in the Swift fork of
`mlx-swift-lm`, validated bit-identical against single-stream output on
all three target model families:

| Model | Architecture | B | Status |
|---|---|---|---|
| Gemma 4 26B-A4B-it-8bit | dense + MoE (4B active) | 2 | ✅ batched ≡ single |
| Qwen 3.5 0.8B-MLX-4bit | hybrid SSM + full attention | 2 | ✅ batched ≡ single |
| Qwen3 0.6B-8bit | dense full attention | 2 / 4 | ✅ batched ≡ single (B=2 / B=4 ragged) |
| Qwen3 0.6B-8bit | dense full attention | 4 | ✅ same-prompt determinism |

This is the same architecture upstream Python `mlx_lm` uses
(`BatchKVCache` + `BatchGenerator` from PR #443) ported to Swift, plus a
hybrid-cache extension (`BatchedCache` protocol) that supports both
full-attention and SSM-style layers in the same model.

## What's in the fork

| File | Purpose |
|---|---|
| `Libraries/MLXLMCommon/BatchKVCache.swift` | Concrete `KVCache + BatchPositionedKVCache + BatchedCache` with per-row offsets, in-place `filter`/`extend`/`extract`/`merge`. |
| `Libraries/MLXLMCommon/SequenceStateMachine.swift` | Per-row trie-based stop-sequence detector (multi-token + state transitions). |
| `Libraries/MLXLMCommon/GenerationBatch.swift` | Decode-phase batch: one forward pass per step, per-row sampler / stop / max_tokens, in-place filter+extend on `[any BatchedCache]`. |
| `Libraries/MLXLMCommon/PromptProcessingBatch.swift` | Prefill-phase batch: chunked prefill, right-padded inputs + `finalize()` rolling on full-attention layers. |
| `Libraries/MLXLMCommon/BatchGenerator.swift` | Orchestrator: `insert` / `next` / `close`. Probes `model.newCache(parameters:)` for per-layer cache topology and allocates a batched analog (BatchKVCache for full attention, MambaCache/ArraysCache for SSM). |
| `Libraries/MLXLMCommon/KVCache.swift` | `createCausalMask` extended with a `leftPadding: MLXArray?` parameter. `BatchedCache` protocol + `ArraysCache` conformance. |
| `Libraries/MLXLLM/Models/Gemma4Text.swift` | Added `Gemma4Router`, `Gemma4Experts` (using `SwitchGLU` with GeGLU activation), MoE config fields, decoder-layer MoE branch with `post_feedforward_layernorm_1` / `pre_feedforward_layernorm_2` / `post_feedforward_layernorm_2`, `sanitize` for fused `gate_up_proj` split. Pre-existing K-eq-V `Attention` bug fixed (was reusing post-RoPE keys for `values`). |

## What's in `provider-swift`

| File | Purpose |
|---|---|
| `Package.swift` | Bumped `swift-transformers` from `0.1.12` to `1.3.0` (so `TokenizersBackend` registers and Qwen 3.5 / Qwen3-VL tokenizers load via `AutoTokenizer`). |
| `Tests/ProviderCoreTests/BatchKVCacheTests.swift` | 12 cache-layer unit tests with synthetic MLXArrays (no model load). |
| `Tests/ProviderCoreTests/SequenceStateMachineTests.swift` | 5 state-machine unit tests. |
| `Tests/ProviderCoreTests/ContinuousBatchingLiveTests.swift` | 5 end-to-end live diff tests gated by `DARKBLOOM_LIVE_MLX_TESTS=1` (Gemma additionally by `DARKBLOOM_LIVE_MLX_GEMMA=1`). |

## Test results

- **Default suite:** 123 / 123 tests pass (12 suites).
- **Live (`DARKBLOOM_LIVE_MLX_TESTS=1`):** Qwen3 + Qwen 3.5 — 4 / 4 batched diff tests pass.
- **Live (`DARKBLOOM_LIVE_MLX_GEMMA=1`):** Gemma 4 26B-A4B (MoE) — 1 / 1 batched diff test passes (5 / 5 total).

## Why batched-vs-single can diverge past N tokens

Greedy batched decode and single-stream decode are bit-identical only
when no token-step has two top logits within float precision of each
other. Larger batch dims change matmul reduction order; bf16/fp16
accumulation can flip the top-1 argmax in close-call cases. vLLM, mlx-lm,
and sglang all exhibit this property.

The diff tests enforce a structural-correctness floor (≥ first 4 tokens
must match) and log further divergence as a non-failing diagnostic. The
**same-prompt determinism** test (4 copies of the same prompt across all
batch positions must produce identical output) is the strictest signal —
it would fail immediately on any cache-leak / mask-leak / per-row-offset
bug.

## Pre-existing bugs found and fixed

`Gemma4Attention` K-eq-V branch (`Gemma4Text.swift`): when `vProj == nil`
and `attentionKeqV: true` (Gemma 4 26B/31B), the Swift code reused `k`
AFTER `kNorm` + transpose + RoPE for `values`, then ran `vNorm` and a
second transpose. That double-transposed `v` to `[B, L, n_kv_heads, D]`
while keys stayed `[B, n_kv_heads, L, D]`, crashing the cache update
with `Shapes (1,28,2,512) and (1,2,28,512) cannot be broadcast`. Fix:
capture pre-norm `kRaw` and use that for the K-eq-V `values` branch,
matching `mlx_lm.gemma4_text.Attention`.

## Pending follow-ups

1. **Wire `BatchGenerator` into `ProviderCore.BatchScheduler`.** Replaces
   per-request `Task` admission-control with a single `BatchGenerator`
   actor handling all concurrent requests. ~150-LoC integration; the
   pieces are all built. Surface API stays the same.

2. **Per-row sampling.** `temperature` / `top-p` / `top-k` / `seed`. The
   `RowSampler` typealias and `[RowSampler?]` plumbing are wired through
   `BatchGenerator` — needs concrete batched samplers.

3. **Throughput benchmarks.** Locked golden numbers for B=1/2/4/8 vs
   single-stream on each of the three model families. Regression-gated.

4. **Eviction + admission diff test.** Row 0 finishes mid-batch, row 5
   takes its slot, assert row 5's output matches a solo run. The only
   batching invariant not directly covered today.

5. **Long soak test.** 1000 prompts through B=4 slots, no memory leak.

## File map for the next session

1. `provider-swift/Sources/ProviderCore/Inference/BatchScheduler.swift` — the actor that needs replacing.
2. `provider-swift/Sources/ProviderCore/ProviderLoop.swift` — `handleInferenceRequest()` is where the new BatchGenerator-backed scheduler will hook in.
3. `libs/mlx-swift-lm/Libraries/MLXLMCommon/BatchGenerator.swift` — `next()` / `admitFromQueue()` / `makeBatchedCache(batchSize:)` are load-bearing.
4. `provider-swift/Tests/ProviderCoreTests/ContinuousBatchingLiveTests.swift` — diff-harness pattern for new sampling / eviction tests.
