// End-to-end performance tests for the Swift provider.
//
// Goal: produce reproducible TTFT (time-to-first-token) and throughput
// numbers that mirror what a coordinator-driven request actually pays
// for in production. Four scenarios, each implemented as a parameterised
// helper and run against two models:
//
//   A) Warm + plaintext  -- pure inference TTFT (baseline)
//   B) Cold (model load) -- load_time + warm_TTFT
//   C) Warm + encrypted  -- encrypt + decrypt + warm_TTFT (full E2E)
//   D) Warm + batched    -- 1, 2, 4 concurrent requests, per-row TTFT
//
//   Qwen3 0.6B-8bit             -- smoke-tier (DARKBLOOM_LIVE_MLX_TESTS=1)
//   Gemma 4 26B-A4B-it-8bit MoE -- production-tier (DARKBLOOM_LIVE_MLX_GEMMA=1)
//
// Numbers print to stderr in a `[perf]` prefix so they're easy to grep
// out of CI logs.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import ProviderCore

@Suite("performance: end-to-end TTFT", .serialized)
struct PerformanceLiveTests {

    // MARK: - Configuration

    /// Static configuration for one performance scenario. Captures the
    /// model ID, the wired-memory budget MLX should target, and the
    /// number of warm-iteration samples to collect (Gemma uses fewer
    /// iterations because each load is ~30 s of disk I/O).
    private struct ModelConfig: Sendable {
        let label: String
        let modelID: String
        let wiredMemoryGB: Int
        let warmIterations: Int
        let coldIterations: Int
        let batchSizes: [Int]
        let maxTokens: Int
    }

    private static let qwen = ModelConfig(
        label: "Qwen3 0.6B-8bit",
        modelID: "mlx-community/Qwen3-0.6B-8bit",
        wiredMemoryGB: 8,
        warmIterations: 3,
        coldIterations: 2,
        batchSizes: [1, 2, 4],
        maxTokens: 16
    )

    private static let gemma = ModelConfig(
        label: "Gemma 4 26B-A4B-it-8bit (MoE)",
        modelID: "mlx-community/gemma-4-26b-a4b-it-8bit",
        wiredMemoryGB: 64,
        warmIterations: 2,
        coldIterations: 1,
        batchSizes: [1, 2, 4],
        maxTokens: 8
    )

    /// Small, ASCII-only prompt -- avoids non-Latin tokenizer detok costs
    /// dominating the warm TTFT measurement.
    private let promptText = "Reply with the single word 'hi'."

    private func sampleRequest(for config: ModelConfig) -> ChatCompletionRequest {
        ChatCompletionRequest(
            model: config.modelID,
            messages: [ChatMessage(role: "user", content: promptText)],
            temperature: 0.0,
            max_tokens: config.maxTokens
        )
    }

    private static var liveEnabled: Bool { LiveInferenceFixtures.liveTestsEnabled }
    private static var gemmaEnabled: Bool { LiveInferenceFixtures.gemmaTestsEnabled }

    // ====================================================================
    // MARK: - Qwen3 0.6B (smoke tier; DARKBLOOM_LIVE_MLX_TESTS=1)
    // ====================================================================

    @Test("warm TTFT baseline (Qwen3 0.6B)", .enabled(if: liveEnabled))
    func qwenWarmTTFT() async throws { try await runWarmTTFT(Self.qwen) }

    @Test("cold TTFT (Qwen3 0.6B)", .enabled(if: liveEnabled))
    func qwenColdTTFT() async throws { try await runColdTTFT(Self.qwen) }

    @Test("encrypted TTFT (Qwen3 0.6B)", .enabled(if: liveEnabled))
    func qwenEncryptedTTFT() async throws { try await runEncryptedTTFT(Self.qwen) }

    @Test("batched TTFT + throughput (Qwen3 0.6B)", .enabled(if: liveEnabled))
    func qwenBatchedTTFT() async throws { try await runBatchedTTFT(Self.qwen) }

    // ====================================================================
    // MARK: - Gemma 4 26B-A4B-it-8bit MoE (production tier;
    //         DARKBLOOM_LIVE_MLX_TESTS=1 + DARKBLOOM_LIVE_MLX_GEMMA=1)
    // ====================================================================

    @Test("warm TTFT baseline (Gemma 26B MoE)", .enabled(if: gemmaEnabled))
    func gemmaWarmTTFT() async throws { try await runWarmTTFT(Self.gemma) }

    @Test("cold TTFT (Gemma 26B MoE)", .enabled(if: gemmaEnabled))
    func gemmaColdTTFT() async throws { try await runColdTTFT(Self.gemma) }

    @Test("encrypted TTFT (Gemma 26B MoE)", .enabled(if: gemmaEnabled))
    func gemmaEncryptedTTFT() async throws { try await runEncryptedTTFT(Self.gemma) }

    @Test("batched TTFT + throughput (Gemma 26B MoE)", .enabled(if: gemmaEnabled))
    func gemmaBatchedTTFT() async throws { try await runBatchedTTFT(Self.gemma) }

    // ====================================================================
    // MARK: - Scenario implementations (parameterised by ModelConfig)
    // ====================================================================

    /// A) Warm baseline -- pure inference TTFT, no encryption, model already loaded.
    private func runWarmTTFT(_ config: ModelConfig) async throws {
        let loaded = try await loadOrSkip(config)
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        // Warm-up pass: the very first generation pays JIT/Metal setup
        // costs that have nothing to do with the steady-state TTFT.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest(for: config))

        var samples: [Duration] = []
        for _ in 0 ..< config.warmIterations {
            samples.append(
                await timeFirstToken(scheduler: scheduler, request: sampleRequest(for: config))
            )
        }
        Self.printRow("\(config.label): warm TTFT", samples: samples, median: median(samples))
        #expect(median(samples) > .zero)
    }

    /// B) Cold -- fresh ModelContainer per iteration so weights are
    /// re-paged from disk. Reports load-only and load+first-token.
    private func runColdTTFT(_ config: ModelConfig) async throws {
        try ensureModelOrSkip(config)
        guard let modelDir = ModelScanner.resolveLocalPath(modelID: config.modelID) else {
            Issue.record("model not in cache")
            return
        }

        var loadSamples: [Duration] = []
        var totalSamples: [Duration] = []

        for _ in 0 ..< config.coldIterations {
            applyMemoryBudget(gigabytes: config.wiredMemoryGB)
            let totalStart = ContinuousClock.now

            let loadStart = ContinuousClock.now
            let container = try await LLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: LocalTokenizerLoader()
            )
            let scheduler = BatchScheduler(
                maxConcurrentRequests: 4,
                pendingTimeout: .seconds(120),
                defaultMaxTokens: 64
            )
            await scheduler.loadModel(container: container, modelId: config.modelID)
            let loadElapsed = ContinuousClock.now - loadStart

            _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest(for: config))
            let totalElapsed = ContinuousClock.now - totalStart

            loadSamples.append(loadElapsed)
            totalSamples.append(totalElapsed)
            await scheduler.unloadModel()
        }

        Self.printRow("\(config.label): cold load",         samples: loadSamples,  median: median(loadSamples))
        Self.printRow("\(config.label): cold load + first", samples: totalSamples, median: median(totalSamples))
        #expect(median(loadSamples) > .zero)
        #expect(median(totalSamples) > median(loadSamples))
    }

    /// C) Encrypted -- full E2E pipeline including NaCl box round-trip.
    private func runEncryptedTTFT(_ config: ModelConfig) async throws {
        let loaded = try await loadOrSkip(config)
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        let providerKeys = NodeKeyPair.generate()
        let consumerKeys = NodeKeyPair.generate()
        let providerPubKeyData = Data(base64Encoded: providerKeys.publicKeyBase64)!
        let consumerPubKeyData = Data(base64Encoded: consumerKeys.publicKeyBase64)!

        // Warm-up.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest(for: config))

        var encryptSamples: [Duration] = []
        var decryptSamples: [Duration] = []
        var ttftSamples: [Duration] = []
        var e2eFirstTokenSamples: [Duration] = []

        for _ in 0 ..< config.warmIterations {
            let payload = try JSONEncoder().encode(sampleRequest(for: config))

            let encStart = ContinuousClock.now
            let ciphertext = try consumerKeys.encrypt(
                recipientPublicKey: providerPubKeyData,
                plaintext: payload
            )
            let encElapsed = ContinuousClock.now - encStart

            let decStart = ContinuousClock.now
            let decrypted = try providerKeys.decrypt(
                senderPublicKey: consumerPubKeyData,
                ciphertext: ciphertext
            )
            let decElapsed = ContinuousClock.now - decStart

            #expect(decrypted == payload, "encrypt/decrypt round-trip must preserve bytes")

            let parsedRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: decrypted)
            let ttft = await timeFirstToken(scheduler: scheduler, request: parsedRequest)

            encryptSamples.append(encElapsed)
            decryptSamples.append(decElapsed)
            ttftSamples.append(ttft)
            e2eFirstTokenSamples.append(encElapsed + decElapsed + ttft)
        }

        Self.printRow("\(config.label): encrypt",                  samples: encryptSamples,        median: median(encryptSamples))
        Self.printRow("\(config.label): decrypt",                  samples: decryptSamples,        median: median(decryptSamples))
        Self.printRow("\(config.label): warm TTFT",                samples: ttftSamples,           median: median(ttftSamples))
        Self.printRow("\(config.label): E2E first-token",          samples: e2eFirstTokenSamples,  median: median(e2eFirstTokenSamples))
        #expect(median(encryptSamples) > .zero)
        #expect(median(decryptSamples) > .zero)
    }

    /// D) Batched -- B=1, B=2, B=4 concurrent submissions on a single
    /// shared scheduler. Reports per-row TTFT and aggregate throughput
    /// (the headline continuous-batching metric).
    private func runBatchedTTFT(_ config: ModelConfig) async throws {
        let loaded = try await LiveInferenceFixtures.loadScheduler(
            modelID: config.modelID,
            maxConcurrentRequests: 4
        )
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        // Warm-up.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest(for: config))

        for batchSize in config.batchSizes {
            let result = await measureBatch(
                scheduler: scheduler,
                batchSize: batchSize,
                request: sampleRequest(for: config)
            )
            Self.printRow(
                "\(config.label): B=\(batchSize) per-row TTFT",
                samples: result.ttft,
                median: median(result.ttft)
            )
            let totalSeconds = Double(result.totalElapsed.components.seconds)
                + Double(result.totalElapsed.components.attoseconds) / 1e18
            let aggregateTPS = totalSeconds > 0
                ? Double(result.totalCompletionTokens) / totalSeconds : 0
            FileHandle.standardError.write(Data(
                "[perf] \(config.label): B=\(batchSize) aggregate throughput  \(String(format: "%.1f", aggregateTPS)) tok/s (\(result.totalCompletionTokens) tokens / \(String(format: "%.2f", totalSeconds))s, \(batchSize) rows)\n".utf8
            ))
            #expect(result.ttft.allSatisfy { $0 > .zero })
        }
    }

    // MARK: - Helpers

    private func loadOrSkip(_ config: ModelConfig) async throws -> (
        scheduler: BatchScheduler,
        container: ModelContainer,
        modelDirectory: URL
    ) {
        applyMemoryBudget(gigabytes: config.wiredMemoryGB)
        do {
            return try await LiveInferenceFixtures.loadScheduler(modelID: config.modelID)
        } catch let skip as LiveFixtureSkip {
            Issue.record("skipped: \(skip.description)")
            throw skip
        }
    }

    private func ensureModelOrSkip(_ config: ModelConfig) throws {
        guard LiveInferenceFixtures.ensureMetallibColocated() != nil else {
            Issue.record("metallib not found; run scripts/fetch-metallib.sh debug")
            return
        }
        guard ModelScanner.resolveLocalPath(modelID: config.modelID) != nil else {
            Issue.record("model '\(config.modelID)' not in cache")
            return
        }
    }

    private func applyMemoryBudget(gigabytes: Int) {
        MLX.GPU.set(memoryLimit: gigabytes * 1024 * 1024 * 1024)
    }

    /// Submit `request` to `scheduler`, measure wall-clock from
    /// submit() to the first `.chunk`, then drain the rest so the
    /// scheduler's row count returns to zero before the next iteration.
    private func timeFirstToken(
        scheduler: BatchScheduler,
        request: ChatCompletionRequest
    ) async -> Duration {
        let start = ContinuousClock.now
        let stream = await scheduler.submit(request: request)
        var ttft: Duration = .zero
        var sawFirst = false
        for await event in stream {
            switch event {
            case .chunk:
                if !sawFirst {
                    ttft = ContinuousClock.now - start
                    sawFirst = true
                }
            case .info, .error:
                break
            }
        }
        return ttft
    }

    private struct BatchResult: Sendable {
        let ttft: [Duration]
        let totalCompletionTokens: Int
        let totalElapsed: Duration
    }

    private func measureBatch(
        scheduler: BatchScheduler,
        batchSize: Int,
        request: ChatCompletionRequest
    ) async -> BatchResult {
        let start = ContinuousClock.now
        struct RowResult: Sendable {
            let ttft: Duration
            let completionTokens: Int
        }
        let rows: [RowResult] = await withTaskGroup(of: RowResult.self) { group in
            for _ in 0 ..< batchSize {
                group.addTask {
                    let stream = await scheduler.submit(request: request)
                    var ttft: Duration = .zero
                    var completionTokens = 0
                    var sawFirst = false
                    for await event in stream {
                        switch event {
                        case .chunk:
                            if !sawFirst {
                                ttft = ContinuousClock.now - start
                                sawFirst = true
                            }
                        case .info(_, let completion, _):
                            completionTokens = completion
                        case .error:
                            break
                        }
                    }
                    return RowResult(ttft: ttft, completionTokens: completionTokens)
                }
            }
            var collected: [RowResult] = []
            for await row in group { collected.append(row) }
            return collected
        }
        let totalElapsed = ContinuousClock.now - start
        return BatchResult(
            ttft: rows.map(\.ttft),
            totalCompletionTokens: rows.reduce(0) { $0 + $1.completionTokens },
            totalElapsed: totalElapsed
        )
    }

    private func median(_ xs: [Duration]) -> Duration {
        guard !xs.isEmpty else { return .zero }
        return xs.sorted()[xs.count / 2]
    }

    // MARK: - Reporting

    private static func printRow(_ name: String, samples: [Duration], median: Duration) {
        let label = name.padding(toLength: 56, withPad: " ", startingAt: 0)
        let cells = samples.map { format($0) }.joined(separator: ", ")
        let line = "[perf] \(label)  median=\(format(median))  samples=[\(cells)]"
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private static func format(_ duration: Duration) -> String {
        let nanos = Double(duration.components.attoseconds) / 1e9
            + Double(duration.components.seconds) * 1e9
        let ms = nanos / 1_000_000.0
        if ms < 10 {
            return String(format: "%.2f ms", ms)
        }
        if ms < 1000 {
            return String(format: "%.1f ms", ms)
        }
        return String(format: "%.2f s", ms / 1000.0)
    }
}
