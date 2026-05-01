import Foundation

// MARK: - Request Types

public struct ChatCompletionRequest: Codable, Sendable {
    public let model: String
    public let messages: [ChatMessage]
    public let temperature: Float?
    public let top_p: Float?
    public let top_k: Int?
    public let max_tokens: Int?
    public let repetition_penalty: Float?
    public let presence_penalty: Float?
    public let frequency_penalty: Float?
    public let stream: Bool?

    public init(
        model: String,
        messages: [ChatMessage],
        temperature: Float? = nil,
        top_p: Float? = nil,
        top_k: Int? = nil,
        max_tokens: Int? = nil,
        repetition_penalty: Float? = nil,
        presence_penalty: Float? = nil,
        frequency_penalty: Float? = nil,
        stream: Bool? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stream = stream
    }
}

public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

// MARK: - Response Types (Streaming)

public struct ChatCompletionChunk: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [ChunkChoice]
    public let usage: ChunkUsage?

    public init(
        id: String,
        object: String = "chat.completion.chunk",
        created: Int,
        model: String,
        choices: [ChunkChoice],
        usage: ChunkUsage? = nil
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct ChunkChoice: Codable, Sendable {
    public let index: Int
    public let delta: ChunkDelta
    public let finish_reason: String?

    public init(index: Int, delta: ChunkDelta, finish_reason: String? = nil) {
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason
    }
}

public struct ChunkDelta: Codable, Sendable {
    public let role: String?
    public let content: String?

    public init(role: String? = nil, content: String? = nil) {
        self.role = role
        self.content = content
    }
}

public struct ChunkUsage: Codable, Sendable {
    public let prompt_tokens: Int
    public let completion_tokens: Int
    public let total_tokens: Int

    public init(prompt_tokens: Int, completion_tokens: Int) {
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
    }
}

// MARK: - Response Types (Non-Streaming)

public struct ChatCompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [ResponseChoice]
    public let usage: ChunkUsage

    public init(
        id: String,
        object: String = "chat.completion",
        created: Int,
        model: String,
        choices: [ResponseChoice],
        usage: ChunkUsage
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct ResponseChoice: Codable, Sendable {
    public let index: Int
    public let message: ResponseMessage
    public let finish_reason: String

    public init(index: Int, message: ResponseMessage, finish_reason: String) {
        self.index = index
        self.message = message
        self.finish_reason = finish_reason
    }
}

public struct ResponseMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String = "assistant", content: String) {
        self.role = role
        self.content = content
    }
}

// MARK: - SSE Chunk Wrapper

public struct SSEChunk: Sendable {
    public let data: String

    public init(data: String) {
        self.data = data
    }

    public var formatted: String {
        "data: \(data)\n\n"
    }

    public static let done = SSEChunk(data: "[DONE]")
}

// MARK: - Errors

public enum InferenceError: Error, Sendable {
    case noModelLoaded
    case modelLoadFailed(String)
    case generationFailed(String)
    case invalidModelDirectory(String)
    case unsupportedRole(String)
}
