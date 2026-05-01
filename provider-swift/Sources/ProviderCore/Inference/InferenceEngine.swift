import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor InferenceEngine {
    private var container: ModelContainer?
    private var loadedModelName: String?
    private var idleTask: Task<Void, Never>?
    private let idleTimeout: Duration
    private let formatter = OpenAIFormatter()

    // Tracks the last time a generate call completed so the idle timer
    // knows when to unload. Updated inside the actor so no lock needed.
    private var lastActivity: ContinuousClock.Instant

    public init(idleTimeout: Duration = .seconds(3600)) {
        self.idleTimeout = idleTimeout
        self.lastActivity = .now
    }

    // MARK: - Model Lifecycle

    public func loadModel(from directory: URL, name: String) async throws {
        if loadedModelName == name, container != nil {
            touchActivity()
            return
        }

        unloadModelSync()

        let loaded = try await LLMModelFactory.shared.loadContainer(
            from: directory,
            using: LocalTokenizerLoader()
        )

        self.container = loaded
        self.loadedModelName = name
        touchActivity()
        scheduleIdleUnload()
    }

    public func unloadModel() {
        unloadModelSync()
    }

    public var isModelLoaded: Bool {
        container != nil
    }

    public var currentModelName: String? {
        loadedModelName
    }

    // MARK: - Streaming Generation

    public func generate(
        request: ChatCompletionRequest
    ) throws -> AsyncStream<SSEChunk> {
        guard let container else {
            throw InferenceError.noModelLoaded
        }

        let rawMessages = try Self.buildRawMessages(request.messages)
        let params = buildParameters(from: request)
        let modelName = request.model
        let completionID = formatter.makeCompletionID()
        let created = Int(Date().timeIntervalSince1970)
        let fmt = formatter

        let (stream, continuation) = AsyncStream<SSEChunk>.makeStream()

        let generationTask = Task.detached {
            defer { continuation.finish() }

            do {
                let generationStream: AsyncStream<Generation> = try await container.perform {
                    context in
                    let input = try await context.processor.prepare(
                        input: UserInput(messages: rawMessages))
                    return try MLXLMCommon.generate(
                        input: input, parameters: params, context: context)
                }

                continuation.yield(
                    fmt.roleChunk(id: completionID, model: modelName, created: created)
                )

                var completionTokens = 0
                var promptTokens = 0
                var stopReason: GenerateStopReason = .stop

                for await generation in generationStream {
                    if Task.isCancelled { break }

                    switch generation {
                    case .chunk(let text):
                        completionTokens += 1
                        continuation.yield(
                            fmt.contentChunk(
                                id: completionID,
                                model: modelName,
                                created: created,
                                text: text
                            )
                        )

                    case .info(let info):
                        promptTokens = info.promptTokenCount
                        completionTokens = info.generationTokenCount
                        stopReason = info.stopReason

                    case .toolCall:
                        break
                    }
                }

                let finishReason = fmt.finishReasonString(stopReason)
                let usage = ChunkUsage(
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens
                )

                continuation.yield(
                    fmt.stopChunk(
                        id: completionID,
                        model: modelName,
                        created: created,
                        finishReason: finishReason,
                        usage: usage
                    )
                )
                continuation.yield(.done)

            } catch {
                continuation.yield(.done)
            }
        }

        continuation.onTermination = { @Sendable _ in
            generationTask.cancel()
        }

        touchActivity()
        return stream
    }

    // MARK: - Non-Streaming Generation

    public func generateFull(
        request: ChatCompletionRequest
    ) async throws -> ChatCompletionResponse {
        guard let container else {
            throw InferenceError.noModelLoaded
        }

        let rawMessages = try Self.buildRawMessages(request.messages)
        let params = buildParameters(from: request)
        let completionID = formatter.makeCompletionID()
        let created = Int(Date().timeIntervalSince1970)

        let generationStream: AsyncStream<Generation> = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(messages: rawMessages))
            return try MLXLMCommon.generate(input: input, parameters: params, context: context)
        }

        var fullContent = ""
        var promptTokens = 0
        var completionTokens = 0
        var stopReason: GenerateStopReason = .stop

        for await generation in generationStream {
            switch generation {
            case .chunk(let text):
                fullContent += text

            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                stopReason = info.stopReason

            case .toolCall:
                break
            }
        }

        touchActivity()

        let finishReason = formatter.finishReasonString(stopReason)
        let usage = ChunkUsage(
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens
        )

        return formatter.nonStreamingResponse(
            id: completionID,
            model: request.model,
            created: created,
            content: fullContent,
            finishReason: finishReason,
            usage: usage
        )
    }

    // MARK: - Idle Management

    private func touchActivity() {
        lastActivity = .now
        scheduleIdleUnload()
    }

    private func scheduleIdleUnload() {
        idleTask?.cancel()
        idleTask = Task { [idleTimeout] in
            while !Task.isCancelled {
                let elapsed = ContinuousClock.now - self.lastActivity
                let remaining = idleTimeout - elapsed
                if remaining <= .zero {
                    self.unloadModelSync()
                    return
                }
                try? await Task.sleep(for: remaining)
            }
        }
    }

    // MARK: - Internals

    private func unloadModelSync() {
        idleTask?.cancel()
        idleTask = nil
        container = nil
        loadedModelName = nil
    }

    private func buildParameters(from request: ChatCompletionRequest) -> GenerateParameters {
        GenerateParameters(
            maxTokens: request.max_tokens,
            temperature: request.temperature ?? 0.6,
            topP: request.top_p ?? 1.0,
            topK: request.top_k ?? 0,
            repetitionPenalty: request.repetition_penalty,
            presencePenalty: request.presence_penalty,
            frequencyPenalty: request.frequency_penalty
        )
    }

    /// Converts OpenAI ChatMessage array to raw Message dictionaries ([String: any Sendable]).
    /// This format is Sendable and goes through the tokenizer's applyChatTemplate.
    private static func buildRawMessages(
        _ messages: [ChatMessage]
    ) throws -> [MLXLMCommon.Message] {
        let validRoles: Set<String> = ["system", "user", "assistant", "tool"]
        return try messages.map { msg in
            guard validRoles.contains(msg.role) else {
                throw InferenceError.unsupportedRole(msg.role)
            }
            return ["role": msg.role, "content": msg.content] as MLXLMCommon.Message
        }
    }
}

// MARK: - Tokenizer Loading

/// Bridges swift-transformers' AutoTokenizer to MLXLMCommon.Tokenizer.
/// This mirrors the exact bridge that mlx-swift-lm's #adaptHuggingFaceTokenizer
/// macro expands to, but done manually since we load from local directories
/// without the HuggingFace Hub client.
struct LocalTokenizerLoader: TokenizerLoader, Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}

private struct TokenizerBridge: @unchecked Sendable, MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
