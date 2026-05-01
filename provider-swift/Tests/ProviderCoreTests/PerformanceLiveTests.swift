// End-to-end performance tests for the Swift provider.
//
// Goal: produce reproducible TTFT (time-to-first-token) and throughput
// numbers that mirror what a coordinator-driven request actually pays
// for in production. Four scenarios:
//
//   A) Warm + plaintext  -- pure inference TTFT (baseline)
//   B) Cold (model load) -- load_time + warm_TTFT
//   C) Warm + encrypted  -- encrypt + decrypt + warm_TTFT (full E2E)
//   D) Warm + batched    -- 1, 2, 4 concurrent requests, per-row TTFT
//
// All measurements use Qwen3 0.6B-8bit so the suite runs in a few
// seconds. Numbers are printed to stdout in a human-readable table;
// tests assert lower-bound liveness (TTFT > 0, completes within a
// generous budget) but don't pin absolute latencies, since those vary
// by hardware.
//
// Gated by DARKBLOOM_LIVE_MLX_TESTS=1.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
@testable import ProviderCore

@Suite("performance: end-to-end TTFT", .serialized)
struct PerformanceLiveTests {

    // MARK: - Shared setup

    /// Small, ASCII-only prompt -- avoids non-Latin tokenizer detok costs
    /// dominating the warm TTFT measurement.
    private let promptText = "Reply with the single word 'hi'."
    private let modelID = LiveInferenceFixtures.tinyModelID

    private var sampleRequest: ChatCompletionRequest {
        ChatCompletionRequest(
            model: modelID,
            messages: [ChatMessage(role: "user", content: promptText)],
            temperature: 0.0,
            max_tokens: 16
        )
    }

    // MARK: - A) Warm baseline

    @Test(
        "warm TTFT baseline (no encryption, no load)",
        .enabled(if: LiveInferenceFixtures.liveTestsEnabled)
    )
    func warmTTFTBaseline() async throws {
        let loaded = try await loadOrSkip()
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        // Warm-up pass: the very first generation pays JIT/Metal setup
        // costs that have nothing to do with the steady-state TTFT.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest)

        var samples: [Duration] = []
        for _ in 0 ..< 3 {
            let ttft = await timeFirstToken(scheduler: scheduler, request: sampleRequest)
            samples.append(ttft)
        }
        let median = samples.sorted()[samples.count / 2]
        Self.report(name: "warm TTFT baseline", samples: samples, median: median)
        #expect(median > .zero)
    }

    // MARK: - B) Cold: model load + first token

    @Test(
        "cold TTFT (model load + first token)",
        .enabled(if: LiveInferenceFixtures.liveTestsEnabled)
    )
    func coldTTFT() async throws {
        try ensureModelOrSkip()

        guard let modelDir = ModelScanner.resolveLocalPath(modelID: modelID) else {
            Issue.record("model not in cache")
            return
        }

        var loadSamples: [Duration] = []
        var totalSamples: [Duration] = []

        for _ in 0 ..< 2 {
            // Fresh container + scheduler each iteration so the model
            // weights are re-paged from disk -- this is the real
            // cold-start path.
            LiveInferenceFixtures.applyMemoryBudget()
            let totalStart = ContinuousClock.now

            let loadStart = ContinuousClock.now
            let container = try await LLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: LocalTokenizerLoader()
            )
            let scheduler = BatchScheduler(
                maxConcurrentRequests: 4,
                pendingTimeout: .seconds(60),
                defaultMaxTokens: 64
            )
            await scheduler.loadModel(container: container, modelId: modelID)
            let loadElapsed = ContinuousClock.now - loadStart

            let ttft = await timeFirstToken(scheduler: scheduler, request: sampleRequest)
            let totalElapsed = ContinuousClock.now - totalStart

            loadSamples.append(loadElapsed)
            totalSamples.append(totalElapsed)
            await scheduler.unloadModel()
            _ = ttft  // included in totalElapsed
        }

        let medianLoad = loadSamples.sorted()[loadSamples.count / 2]
        let medianTotal = totalSamples.sorted()[totalSamples.count / 2]
        Self.printRow("cold load time", samples: loadSamples, median: medianLoad)
        Self.printRow("cold load + first token", samples: totalSamples, median: medianTotal)
        #expect(medianLoad > .zero)
        #expect(medianTotal > medianLoad)
    }

    // MARK: - C) End-to-end with NaCl-box encryption

    @Test(
        "encrypted TTFT (encrypt + decrypt + first token, full E2E)",
        .enabled(if: LiveInferenceFixtures.liveTestsEnabled)
    )
    func encryptedTTFT() async throws {
        let loaded = try await loadOrSkip()
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        let providerKeys = NodeKeyPair.generate()
        let consumerKeys = NodeKeyPair.generate()
        let providerPubKeyData = Data(base64Encoded: providerKeys.publicKeyBase64)!
        let consumerPubKeyData = Data(base64Encoded: consumerKeys.publicKeyBase64)!

        // Warm-up.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest)

        var encryptSamples: [Duration] = []
        var decryptSamples: [Duration] = []
        var ttftSamples: [Duration] = []
        var e2eFirstTokenSamples: [Duration] = []

        for _ in 0 ..< 3 {
            let payload = try JSONEncoder().encode(sampleRequest)

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

        Self.printRow("encrypt (consumer side)",          samples: encryptSamples,        median: median(encryptSamples))
        Self.printRow("decrypt (provider side)",          samples: decryptSamples,        median: median(decryptSamples))
        Self.printRow("inference TTFT (warm)",            samples: ttftSamples,           median: median(ttftSamples))
        Self.printRow("E2E first-token (enc+dec+TTFT)",   samples: e2eFirstTokenSamples,  median: median(e2eFirstTokenSamples))
        #expect(median(encryptSamples) > .zero)
        #expect(median(decryptSamples) > .zero)
    }

    // MARK: - D) Batched concurrent submissions

    @Test(
        "batched TTFT + throughput at B=1, B=2, B=4 (continuous batching)",
        .enabled(if: LiveInferenceFixtures.liveTestsEnabled)
    )
    func batchedTTFT() async throws {
        let loaded = try await LiveInferenceFixtures.loadScheduler(
            modelID: modelID,
            maxConcurrentRequests: 4
        )
        let scheduler = loaded.scheduler
        defer { Task { await scheduler.unloadModel() } }

        // Warm-up.
        _ = await timeFirstToken(scheduler: scheduler, request: sampleRequest)

        for batchSize in [1, 2, 4] {
            let result = await measureBatch(
                scheduler: scheduler,
                batchSize: batchSize,
                request: sampleRequest
            )
            Self.printRow(
                "B=\(batchSize) per-request TTFT",
                samples: result.ttft,
                median: median(result.ttft)
            )
            // Aggregate throughput: total tokens generated across all rows
            // divided by wall-clock from first submit to last completion.
            let totalSeconds = Double(result.totalElapsed.components.seconds)
                + Double(result.totalElapsed.components.attoseconds) / 1e18
            let aggregateTPS = totalSeconds > 0
                ? Double(result.totalCompletionTokens) / totalSeconds : 0
            FileHandle.standardError.write(Data(
                "[perf] B=\(batchSize) aggregate throughput                 \(String(format: "%.1f", aggregateTPS)) tok/s (across all \(batchSize) rows)\n".utf8
            ))
            #expect(result.ttft.allSatisfy { $0 > .zero })
        }
    }

    // MARK: - Helpers

    private func loadOrSkip() async throws -> (
        scheduler: BatchScheduler,
        container: ModelContainer,
        modelDirectory: URL
    ) {
        do {
            return try await LiveInferenceFixtures.loadScheduler(modelID: modelID)
        } catch let skip as LiveFixtureSkip {
            Issue.record("skipped: \(skip.description)")
            throw skip
        }
    }

    private func ensureModelOrSkip() throws {
        guard LiveInferenceFixtures.ensureMetallibColocated() != nil else {
            Issue.record("metallib not found; run scripts/fetch-metallib.sh debug")
            return
        }
        guard ModelScanner.resolveLocalPath(modelID: modelID) != nil else {
            Issue.record("model '\(modelID)' not in cache")
            return
        }
    }

    /// Submit `request` to `scheduler`, measure the wall-clock time from
    /// submit() to the first `.chunk` event. Drains the rest of the
    /// stream so the scheduler's row count returns to zero before the
    /// next iteration.
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

    /// Per-row TTFT and aggregate throughput from a batched submission.
    private struct BatchResult: Sendable {
        let ttft: [Duration]
        let totalCompletionTokens: Int
        let totalElapsed: Duration
    }

    /// Submit `batchSize` identical requests at once.
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

    private static func report(name: String, samples: [Duration], median: Duration) {
        printRow(name, samples: samples, median: median)
    }

    private static func printRow(_ name: String, samples: [Duration], median: Duration) {
        // `%s` would expect a C string; pad the Swift String manually.
        let label = name.padding(toLength: 44, withPad: " ", startingAt: 0)
        let cells = samples.map { format($0) }.joined(separator: ", ")
        let line = "[perf] \(label)  median=\(format(median))  samples=[\(cells)]"
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private static func format(_ duration: Duration) -> String {
        // Convert to milliseconds with one decimal place.
        let nanos = Double(duration.components.attoseconds) / 1e9
            + Double(duration.components.seconds) * 1e9
        let ms = nanos / 1_000_000.0
        if ms < 10 {
            return String(format: "%.2f ms", ms)
        }
        return String(format: "%.1f ms", ms)
    }
}

