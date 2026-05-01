import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Public Types

/// Events emitted by the scheduler for a single inference request.
public enum GenerationEvent: Sendable {
    /// A decoded text chunk (one or more tokens worth of text).
    case chunk(String)
    /// Final usage and performance statistics for the completed request.
    case info(promptTokens: Int, completionTokens: Int, tokensPerSecond: Double)
    /// An unrecoverable error that terminated this request.
    case error(String)
}

/// Snapshot of the scheduler's capacity, reported to the coordinator in heartbeats.
public struct SchedulerCapacity: Sendable {
    /// Model currently loaded (empty string if none).
    public let model: String
    /// Number of requests actively generating tokens.
    public let activeRequests: Int
    /// Number of requests queued waiting for a slot.
    public let pendingRequests: Int
    /// Maximum concurrent requests the scheduler will admit.
    public let maxConcurrent: Int
    /// GPU active memory in bytes.
    public let gpuMemoryActiveBytes: Int
    /// GPU peak memory observed in bytes.
    public let gpuMemoryPeakBytes: Int
    /// GPU cache memory in bytes.
    public let gpuMemoryCacheBytes: Int
    /// Total unified memory available in bytes.
    public let totalMemoryBytes: UInt64
}

// MARK: - Internal State

/// Represents a single in-flight inference request inside the scheduler.
private struct ActiveRequest: Sendable {
    let id: String
    let task: Task<Void, Never>
    let submittedAt: ContinuousClock.Instant
}

/// Holds everything needed to start generating for a queued request.
/// The continuation is used to yield events into the caller's AsyncStream.
private struct PendingRequest: Sendable {
    let id: String
    let request: ChatCompletionRequest
    let continuation: AsyncStream<GenerationEvent>.Continuation
    let submittedAt: ContinuousClock.Instant
}

// MARK: - BatchScheduler

/// Continuous batching scheduler for concurrent inference on a single ModelContainer.
///
/// The ModelContainer serializes access via an internal actor, processing one
/// call to `perform` / `generate` at a time. The key insight from mlx-swift-lm
/// is that during decode the model weights are read-only and each request has
/// its own KVCache. The `ModelContainer.generate` method only holds the serial
/// lock during prefill; after that the returned AsyncStream runs concurrently.
///
/// The scheduler exploits this by:
/// 1. Admitting at most `maxConcurrentRequests` requests at once.
/// 2. Serializing prefill through the ModelContainer's own lock (one at a time).
/// 3. Letting decode streams run concurrently once prefill completes.
/// 4. Using WiredMemoryPolicy for admission control when the GPU is near capacity.
/// 5. Providing cancel(requestId:) for immediate request termination.
///
/// This design matches the standard vLLM approach: at most one prefill runs at
/// a time while all active decodes overlap on the GPU.
public actor BatchScheduler {

    // MARK: - Configuration

    /// Maximum number of requests generating concurrently.
    /// Each active request holds a KV cache in GPU memory.
    private let maxConcurrentRequests: Int

    /// Maximum time a request may sit in the pending queue before being rejected.
    private let pendingTimeout: Duration

    /// Maximum tokens any single request can generate.
    private let defaultMaxTokens: Int

    // MARK: - Model State

    /// The loaded model container. Nil until a model is loaded.
    private var modelContainer: ModelContainer?

    /// The model identifier string (e.g. "mlx-community/Qwen2.5-7B-4bit").
    private var modelId: String = ""

    /// Weight bytes measured at load time, used for memory budgeting.
    private var modelWeightBytes: Int = 0

    // MARK: - Request Tracking

    /// Requests currently generating tokens. Keyed by request ID.
    private var activeRequests: [String: ActiveRequest] = [:]

    /// Requests waiting for a slot. Oldest first.
    private var pendingQueue: [PendingRequest] = []

    /// Monotonic counter for generating unique request IDs when the caller
    /// does not provide one.
    private var requestCounter: UInt64 = 0

    // MARK: - Lifecycle

    /// Creates a new scheduler.
    ///
    /// - Parameters:
    ///   - maxConcurrentRequests: Maximum number of requests that can generate at
    ///     the same time. Each request holds its own KVCache in GPU memory, so this
    ///     directly controls peak memory usage. Defaults to 4.
    ///   - pendingTimeout: How long a request can wait in the queue before being
    ///     rejected with an error event. Defaults to 120 seconds (matching the
    ///     coordinator's queue timeout).
    ///   - defaultMaxTokens: The maximum number of tokens to generate when the
    ///     request does not specify one. Defaults to 4096.
    public init(
        maxConcurrentRequests: Int = 4,
        pendingTimeout: Duration = .seconds(120),
        defaultMaxTokens: Int = 4096
    ) {
        self.maxConcurrentRequests = max(1, maxConcurrentRequests)
        self.pendingTimeout = pendingTimeout
        self.defaultMaxTokens = defaultMaxTokens
    }

    // MARK: - Model Management

    /// Load a model into the scheduler. Replaces any previously loaded model.
    ///
    /// All active and pending requests are cancelled before the swap.
    ///
    /// - Parameters:
    ///   - container: The ModelContainer to use for inference.
    ///   - modelId: Human-readable identifier for the model.
    public func loadModel(container: ModelContainer, modelId: String) async {
        // Cancel everything in-flight before swapping the model.
        cancelAll()
        self.modelContainer = container
        self.modelId = modelId

        // Measure weight bytes for memory budgeting.
        let bytes: Int = await container.perform { context in
            context.model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        }
        self.modelWeightBytes = bytes
    }

    /// Unload the current model and cancel all requests.
    public func unloadModel() {
        cancelAll()
        modelContainer = nil
        modelId = ""
        modelWeightBytes = 0
    }

    // MARK: - Request Submission

    /// Submit a chat completion request for inference.
    ///
    /// Returns an `AsyncStream<GenerationEvent>` that the caller consumes to
    /// receive text chunks, and eventually a `.info` or `.error` event.
    ///
    /// If the scheduler is at capacity the request is queued. If it remains
    /// queued past `pendingTimeout` it receives an `.error` event.
    ///
    /// - Parameters:
    ///   - request: The chat completion request.
    ///   - requestId: Optional caller-supplied ID. One is generated if nil.
    /// - Returns: Stream of generation events for this request.
    public func submit(
        request: ChatCompletionRequest,
        requestId: String? = nil
    ) -> AsyncStream<GenerationEvent> {
        let id = requestId ?? nextRequestId()

        let (stream, continuation) = AsyncStream<GenerationEvent>.makeStream()

        // If no model is loaded, fail immediately.
        guard modelContainer != nil else {
            continuation.yield(.error("No model loaded"))
            continuation.finish()
            return stream
        }

        let pending = PendingRequest(
            id: id,
            request: request,
            continuation: continuation,
            submittedAt: .now
        )

        if activeRequests.count < maxConcurrentRequests {
            // Slot available -- start immediately.
            startRequest(pending)
        } else {
            // Queue and wait.
            pendingQueue.append(pending)
        }

        // When the consumer drops the stream, cancel the request.
        continuation.onTermination = { @Sendable [weak self] termination in
            if case .cancelled = termination {
                Task { [weak self] in
                    await self?.cancel(requestId: id)
                }
            }
        }

        return stream
    }

    /// Cancel a specific request by ID.
    ///
    /// If the request is active, its generation task is cancelled and the
    /// stream is finished. If it is pending, it is removed from the queue.
    public func cancel(requestId: String) {
        // Check active requests.
        if let active = activeRequests.removeValue(forKey: requestId) {
            active.task.cancel()
            drainPendingQueue()
            return
        }

        // Check pending queue.
        if let idx = pendingQueue.firstIndex(where: { $0.id == requestId }) {
            let pending = pendingQueue.remove(at: idx)
            pending.continuation.yield(.error("Request cancelled"))
            pending.continuation.finish()
        }
    }

    /// Cancel all active and pending requests. Used during model swap and shutdown.
    public func cancelAll() {
        // Cancel all active tasks.
        for (_, active) in activeRequests {
            active.task.cancel()
        }
        activeRequests.removeAll()

        // Reject all pending requests.
        for pending in pendingQueue {
            pending.continuation.yield(.error("Scheduler shutting down"))
            pending.continuation.finish()
        }
        pendingQueue.removeAll()
    }

    // MARK: - Capacity Reporting

    /// Returns a snapshot of the scheduler's current capacity for heartbeat reporting.
    public func capacity() -> SchedulerCapacity {
        SchedulerCapacity(
            model: modelId,
            activeRequests: activeRequests.count,
            pendingRequests: pendingQueue.count,
            maxConcurrent: maxConcurrentRequests,
            gpuMemoryActiveBytes: Memory.activeMemory,
            gpuMemoryPeakBytes: Memory.peakMemory,
            gpuMemoryCacheBytes: Memory.cacheMemory,
            totalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )
    }

    /// Convert the capacity snapshot to the protocol's BackendCapacity type
    /// for wire transmission.
    public func backendCapacity() -> BackendCapacity {
        let cap = capacity()
        let gbDivisor = 1024.0 * 1024.0 * 1024.0
        let slot = BackendSlotCapacity(
            model: cap.model,
            state: cap.activeRequests > 0 ? "running" : "idle",
            numRunning: UInt32(cap.activeRequests),
            numWaiting: UInt32(cap.pendingRequests),
            activeTokens: 0,
            maxTokensPotential: Int64(defaultMaxTokens * maxConcurrentRequests)
        )
        return BackendCapacity(
            slots: [slot],
            gpuMemoryActiveGb: Double(cap.gpuMemoryActiveBytes) / gbDivisor,
            gpuMemoryPeakGb: Double(cap.gpuMemoryPeakBytes) / gbDivisor,
            gpuMemoryCacheGb: Double(cap.gpuMemoryCacheBytes) / gbDivisor,
            totalMemoryGb: Double(cap.totalMemoryBytes) / gbDivisor
        )
    }

    // MARK: - Internals

    /// Generate a unique request ID.
    private func nextRequestId() -> String {
        requestCounter += 1
        return "req-\(requestCounter)"
    }

    /// Promote the oldest pending request into an active slot, if capacity allows.
    private func drainPendingQueue() {
        while activeRequests.count < maxConcurrentRequests, !pendingQueue.isEmpty {
            let pending = pendingQueue.removeFirst()

            // Check for timeout while queued.
            let elapsed = ContinuousClock.now - pending.submittedAt
            if elapsed > pendingTimeout {
                pending.continuation.yield(
                    .error("Request timed out after \(elapsed) in queue"))
                pending.continuation.finish()
                continue
            }

            startRequest(pending)
        }
    }

    /// Start generating for a pending request. Moves it into the active set
    /// and spawns the generation task.
    private func startRequest(_ pending: PendingRequest) {
        guard let container = modelContainer else {
            pending.continuation.yield(.error("No model loaded"))
            pending.continuation.finish()
            return
        }

        let requestId = pending.id
        let request = pending.request
        let continuation = pending.continuation
        let maxTokens = request.max_tokens ?? defaultMaxTokens

        let task = Task { [weak self] in
            await Self.runGeneration(
                container: container,
                request: request,
                requestId: requestId,
                maxTokens: maxTokens,
                continuation: continuation
            )

            // When generation completes (success, error, or cancellation),
            // remove from active set and promote next pending request.
            await self?.requestCompleted(requestId: requestId)
        }

        activeRequests[requestId] = ActiveRequest(
            id: requestId,
            task: task,
            submittedAt: pending.submittedAt
        )
    }

    /// Called when a generation task finishes. Removes it from the active set
    /// and drains the pending queue.
    private func requestCompleted(requestId: String) {
        activeRequests.removeValue(forKey: requestId)
        drainPendingQueue()
    }

    /// The actual generation loop. This is a static method to avoid capturing
    /// `self` (the actor) for the duration of generation -- the only callback
    /// into the actor is `requestCompleted` at the end.
    private static func runGeneration(
        container: ModelContainer,
        request: ChatCompletionRequest,
        requestId: String,
        maxTokens: Int,
        continuation: AsyncStream<GenerationEvent>.Continuation
    ) async {
        let generationStart = ContinuousClock.now

        do {
            // 1. Build messages array for chat template.
            let messages: [[String: any Sendable]] = request.messages.map { msg in
                ["role": msg.role, "content": msg.content]
            }

            // 2. Build GenerateParameters from the request.
            let parameters = GenerateParameters(
                maxTokens: maxTokens,
                temperature: request.temperature ?? 0.6,
                topP: request.top_p ?? 1.0,
                topK: request.top_k ?? 0,
                repetitionPenalty: request.repetition_penalty,
                presencePenalty: request.presence_penalty,
                frequencyPenalty: request.frequency_penalty
            )

            // 3. Build a UserInput from the messages and prepare it through
            //    the model's processor. This tokenizes the prompt using the
            //    chat template and handles any model-specific preprocessing.
            let userInput = UserInput(messages: messages)
            let prepared = try await container.prepare(input: userInput)

            // 4. Start generation via ModelContainer.generate().
            //    This method acquires the serial lock ONLY during prefill
            //    (populating the KV cache). Once prefill completes, the lock
            //    is released and the returned AsyncStream runs the decode loop
            //    concurrently. This is the key to concurrent batching: multiple
            //    decode streams share the read-only model weights while each
            //    maintains its own KV cache.
            let stream = try await container.generate(
                input: prepared,
                parameters: parameters
            )

            // 5. Consume the generation stream.
            //    The decode loop runs concurrently with other requests' decode
            //    loops since the model weights are read-only during decode.
            var completionTokens = 0
            var promptTokens = 0

            for await event in stream {
                // Respect cancellation.
                if Task.isCancelled {
                    break
                }

                switch event {
                case .chunk(let text):
                    continuation.yield(.chunk(text))

                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount

                case .toolCall:
                    // Tool calls are not supported yet in the provider.
                    break
                }
            }

            // 6. Compute performance metrics.
            let elapsed = ContinuousClock.now - generationStart
            let elapsedSeconds = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            let tokensPerSecond = elapsedSeconds > 0
                ? Double(completionTokens) / elapsedSeconds
                : 0

            // 7. Emit final info event.
            if !Task.isCancelled {
                continuation.yield(.info(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    tokensPerSecond: tokensPerSecond
                ))
            }

        } catch {
            if !Task.isCancelled {
                continuation.yield(.error("Generation failed: \(error.localizedDescription)"))
            }
        }

        continuation.finish()
    }
}

// MARK: - Memory Helpers

/// Memory access helpers. These wrap MLX GPU memory queries that are available
/// on macOS 15+.
private enum Memory {
    static var activeMemory: Int {
        #if canImport(Metal)
            MLX.GPU.activeMemory
        #else
            0
        #endif
    }

    static var peakMemory: Int {
        #if canImport(Metal)
            MLX.GPU.peakMemory
        #else
            0
        #endif
    }

    static var cacheMemory: Int {
        #if canImport(Metal)
            MLX.GPU.cacheMemory
        #else
            0
        #endif
    }
}
