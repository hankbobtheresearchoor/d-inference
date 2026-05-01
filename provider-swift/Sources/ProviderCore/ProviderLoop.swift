/// ProviderLoop -- the main event loop that ties all subsystems together.
///
/// Owns the CoordinatorClient, BatchScheduler, NodeKeyPair, and
/// SecureEnclaveIdentity. Processes coordinator events: inference requests,
/// cancellations, attestation challenges, and connection lifecycle.
///
/// Each inference request spawns its own Task for concurrent processing.
/// The BatchScheduler manages admission control and model loading.
/// Responses are encrypted with the consumer's ephemeral public key
/// and streamed back through the coordinator.

import CryptoKit
import Foundation
#if canImport(os)
import os
#endif

// MARK: - SendHandle (Sendable wrapper for the coordinator send function)

/// Wraps the coordinator's outbound send function so it can be captured in
/// Tasks and closures that require `Sendable`. The underlying function is
/// thread-safe (it yields into an `AsyncStream.Continuation`) but its type
/// signature from `CoordinatorClient.start()` does not carry `@Sendable`.
public final class SendHandle: @unchecked Sendable {
    private let fn: (OutboundMessage) -> Void

    public init(_ fn: @escaping (OutboundMessage) -> Void) {
        self.fn = fn
    }

    public func send(_ message: OutboundMessage) {
        fn(message)
    }
}

private enum ProviderLoopError: Error, CustomStringConvertible {
    case binaryHashUnavailable

    var description: String {
        switch self {
        case .binaryHashUnavailable:
            return "provider binary hash could not be computed"
        }
    }
}

// MARK: - Configuration

public struct ProviderLoopConfig: Sendable {
    public let coordinatorURL: String
    public let hardware: HardwareInfo
    public let models: [ModelInfo]
    public let config: ProviderConfig
    public let authToken: String?
    public let runtimeHashes: RuntimeHashes?
    public let modelHashes: [String: String]

    public init(
        coordinatorURL: String,
        hardware: HardwareInfo,
        models: [ModelInfo],
        config: ProviderConfig,
        authToken: String? = nil,
        runtimeHashes: RuntimeHashes? = nil,
        modelHashes: [String: String] = [:]
    ) {
        self.coordinatorURL = coordinatorURL
        self.hardware = hardware
        self.models = models
        self.config = config
        self.authToken = authToken
        self.runtimeHashes = runtimeHashes
        self.modelHashes = modelHashes
    }
}

// MARK: - ProviderLoop

public actor ProviderLoop {
    private let loopConfig: ProviderLoopConfig
    private let keyPair: NodeKeyPair
    private let seIdentity: SecureEnclaveIdentity?
    private let attestationBuilder: AttestationBuilder?
    private let scheduler: BatchScheduler
    private let stats: AtomicProviderStats
    private let state: ProviderState
    private let cancellationRegistry: InferenceCancellationRegistry

    /// Tracks in-flight inference tasks by request ID so they can be cancelled.
    private var inflightTasks: [String: Task<Void, Never>] = [:]

    /// Cached security posture from startup verification.
    private var securityPosture: SecurityPosture?

    /// Cached binary hash for attestation responses.
    private var binaryHash: String?

    private let logger = ProviderLogger(subsystem: "dev.darkbloom.provider", category: "loop")

    // MARK: - Initialization

    public init(config: ProviderLoopConfig) throws {
        self.loopConfig = config
        NodeKeyPair.purgeLegacyFiles()
        self.keyPair = NodeKeyPair.generate()
        self.seIdentity = try SecureEnclaveIdentity.createEphemeral()
        self.attestationBuilder = seIdentity.map { AttestationBuilder(identity: $0) }
        self.stats = AtomicProviderStats()
        self.state = ProviderState()
        self.cancellationRegistry = InferenceCancellationRegistry()
        self.scheduler = BatchScheduler(
            maxConcurrentRequests: 4,
            pendingTimeout: .seconds(120),
            defaultMaxTokens: 4096
        )
    }

    // MARK: - Main Run Loop

    public func run() async throws {
        logger.info("darkbloom \(ProviderCore.version) starting")
        logger.info("Hardware: \(loopConfig.hardware.chipName), \(loopConfig.hardware.memoryGb) GB RAM, \(loopConfig.hardware.gpuCores) GPU cores")
        logger.info("Models: \(loopConfig.models.count) advertised")
        logger.info("Coordinator: \(loopConfig.coordinatorURL)")

        // 1. Apply security hardening
        try await applySecurityHardening()

        // 2. Build attestation blob for registration
        let attestation = buildRegistrationAttestation()

        // 3. Create coordinator client config
        let coordinatorConfig = CoordinatorClientConfig(
            url: loopConfig.coordinatorURL,
            hardware: loopConfig.hardware,
            models: loopConfig.models,
            backendName: "mlx-swift",
            heartbeatInterval: TimeInterval(loopConfig.config.coordinator.heartbeatIntervalSecs),
            publicKey: keyPair.publicKeyBase64,
            walletAddress: nil,
            attestation: attestation,
            authToken: loopConfig.authToken,
            runtimeHashes: loopConfig.runtimeHashes,
            modelHashes: loopConfig.modelHashes,
            privacyCapabilities: privacyCapabilitiesForRegistration()
        )

        // 4. Create coordinator client and start connection
        let coordinator = CoordinatorClient(
            config: coordinatorConfig,
            stats: stats,
            state: state
        )

        let (events, sendFn) = await coordinator.start()
        let send = SendHandle(sendFn)

        logger.info("Coordinator client started, entering event loop")

        // 5. Process events
        for await event in events {
            switch event {
            case .connected:
                logger.info("Connected to coordinator")

            case .disconnected:
                logger.warning("Disconnected from coordinator")
                // Cancel all in-flight requests on disconnect -- the coordinator
                // will not route responses for a dead connection.
                await cancelAllInflight()

            case .inferenceRequest(let requestId, let body, let responsePublicKey):
                await handleInferenceRequest(
                    requestId: requestId,
                    body: body,
                    responsePublicKey: responsePublicKey,
                    send: send
                )

            case .cancel(let requestId):
                await handleCancellation(requestId: requestId)

            case .attestationChallenge(let nonce, let timestamp):
                await handleAttestationChallenge(
                    nonce: nonce,
                    timestamp: timestamp,
                    send: send
                )

            case .runtimeOutdated(let mismatches):
                logger.warning("Runtime outdated: \(mismatches.count) mismatch(es)")
                for m in mismatches {
                    logger.warning("  \(m.component): expected=\(m.expected), got=\(m.got)")
                }
            }
        }

        logger.info("Event stream ended, shutting down")
        await coordinator.shutdown()
        await cancelAllInflight()
    }

    // MARK: - Security Hardening

    private func applySecurityHardening() async throws {
        #if !DEBUG
        let posture = try verifySecurityPosture()
        guard let binaryHash = posture.binaryHash, !binaryHash.isEmpty else {
            logger.error("Security hardening failed: provider binary hash unavailable")
            throw ProviderLoopError.binaryHashUnavailable
        }
        self.securityPosture = posture
        self.binaryHash = binaryHash
        logger.info("Security posture verified: SIP=\(posture.sipEnabled), RDMA_disabled=\(posture.rdmaDisabled), SE=\(SecureEnclave.isAvailable)")
        #else
        logger.info("Security hardening skipped in DEBUG mode")
        self.binaryHash = selfBinaryHash()
        #endif
    }

    private func privacyCapabilitiesForRegistration() -> PrivacyCapabilities {
        if let posture = securityPosture {
            return PrivacyCapabilities(
                textBackendInprocess: true,
                textProxyDisabled: true,
                pythonRuntimeLocked: true,
                dangerousModulesBlocked: true,
                sipEnabled: posture.sipEnabled,
                antiDebugEnabled: posture.antiDebugEnabled,
                coreDumpsDisabled: posture.coreDumpsDisabled,
                envScrubbed: posture.envScrubbed,
                hypervisorActive: false
            )
        }

        return PrivacyCapabilities(
            textBackendInprocess: true,
            textProxyDisabled: true,
            pythonRuntimeLocked: true,
            dangerousModulesBlocked: true,
            sipEnabled: SecurityChecks.isSIPEnabled(),
            antiDebugEnabled: false,
            coreDumpsDisabled: false,
            envScrubbed: false,
            hypervisorActive: SecurityChecks.isHypervisorActive()
        )
    }

    // MARK: - Attestation

    private func buildRegistrationAttestation() -> RawJSON? {
        guard let builder = attestationBuilder else {
            logger.info("No Secure Enclave identity -- registration without attestation")
            return nil
        }
        do {
            let jsonData = try builder.buildAttestationJSON(
                encryptionPublicKey: keyPair.publicKeyBase64,
                binaryHash: binaryHash
            )
            return RawJSON(rawBytes: jsonData)
        } catch {
            logger.error("Failed to build attestation: \(error)")
            return nil
        }
    }

    // MARK: - Inference Request Handling

    private func handleInferenceRequest(
        requestId: String,
        body: Data,
        responsePublicKey: [UInt8]?,
        send: SendHandle
    ) async {
        logger.info("Processing inference request: \(requestId)")

        // 1. Decrypt the request body
        let decryptedData: Data
        do {
            // The body arrives as a base64-encoded ciphertext string from
            // CoordinatorClient.handleIncomingText which wraps
            // encrypted.ciphertext.utf8 into Data. We need to reconstruct
            // the EncryptedPayload to decrypt it properly.
            guard let responseKey = responsePublicKey else {
                logger.error("[\(requestId)] No response public key for encrypted request")
                send.send(.inferenceError(requestId: requestId, error: "missing response public key", statusCode: 400))
                return
            }

            // The body is the base64 ciphertext string. The ephemeral public key
            // was already extracted by CoordinatorClient and passed as responsePublicKey.
            // However, looking at CoordinatorClient.handleIncomingText more carefully:
            // it passes Data(encrypted.ciphertext.utf8) as body and the ephemeralPublicKey
            // decoded as responsePublicKey. The ciphertext is base64, so we need to
            // base64-decode it, then use the ephemeral key to decrypt.
            guard let ciphertextBase64String = String(data: body, encoding: .utf8),
                  let ciphertextData = Data(base64Encoded: ciphertextBase64String) else {
                logger.error("[\(requestId)] Failed to decode ciphertext from base64")
                send.send(.inferenceError(requestId: requestId, error: "invalid ciphertext encoding", statusCode: 400))
                return
            }

            let senderPublicKey = Data(responseKey)
            decryptedData = try keyPair.decrypt(senderPublicKey: senderPublicKey, ciphertext: ciphertextData)
        } catch {
            logger.error("[\(requestId)] Decryption failed: \(error)")
            send.send(.inferenceError(requestId: requestId, error: "decryption failed", statusCode: 400))
            return
        }

        // 2. Parse the chat completion request
        let chatRequest: ChatCompletionRequest
        do {
            chatRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: decryptedData)
        } catch {
            logger.error("[\(requestId)] Failed to parse chat request: \(error)")
            send.send(.inferenceError(requestId: requestId, error: "invalid request body: \(error.localizedDescription)", statusCode: 400))
            return
        }

        // 3. Send inference_accepted
        send.send(.inferenceAccepted(requestId: requestId))

        // 4. Ensure model is loaded
        let modelId = chatRequest.model
        do {
            try await ensureModelLoaded(modelId: modelId)
        } catch {
            logger.error("[\(requestId)] Failed to load model '\(modelId)': \(error)")
            send.send(.inferenceError(requestId: requestId, error: "model load failed: \(error.localizedDescription)", statusCode: 500))
            return
        }

        // 5. Register cancellation token
        let token = await cancellationRegistry.register(requestId: requestId)

        // 6. Capture values for the spawned task
        let responsePublicKeyData: Data? = responsePublicKey.map { Data($0) }
        let kp = self.keyPair
        let sched = self.scheduler
        let providerStats = self.stats
        let providerState = self.state
        let registry = self.cancellationRegistry
        let signingIdentity = self.seIdentity
        let log = self.logger

        // 7. Spawn inference task
        let task = Task.detached {
            defer {
                Task { await registry.finish(requestId: requestId) }
            }

            providerState.inferenceActive = true

            var usageAccumulator = UsageAccumulator()
            var fullResponseText = ""
            let formatter = ChatSSEFormatter()
            let responseId = "chatcmpl-\(UUID().uuidString.prefix(12).lowercased())"
            let created = Int(Date().timeIntervalSince1970)

            let emitSSE: @Sendable (String) -> Void = { sseData in
                var encryptedPayload: EncryptedPayload?
                if let recipientKey = responsePublicKeyData {
                    do {
                        encryptedPayload = try kp.encryptPayload(
                            recipientPublicKey: recipientKey,
                            plaintext: Data(sseData.utf8)
                        )
                    } catch {
                        log.warning("[\(requestId)] Chunk encryption failed: \(error)")
                    }
                }

                send.send(.inferenceChunk(
                    requestId: requestId,
                    data: encryptedPayload != nil ? "" : sseData,
                    encryptedData: encryptedPayload
                ))
            }

            if let roleChunk = try? formatter.roleChunk(
                id: responseId,
                model: chatRequest.model,
                created: created
            ) {
                emitSSE(roleChunk.formatted)
            }

            // Submit to the BatchScheduler
            let generationStream = await sched.submit(
                request: chatRequest,
                requestId: requestId
            )

            for await event in generationStream {
                // Check cancellation
                if token.isCancelled {
                    log.info("[\(requestId)] Cancelled during generation")
                    send.send(.inferenceError(requestId: requestId, error: "request cancelled", statusCode: 499))
                    return
                }

                switch event {
                case .chunk(let text):
                    fullResponseText += text
                    usageAccumulator.recordCompletionChunk()

                    if let contentChunk = try? formatter.contentChunk(
                        id: responseId,
                        model: chatRequest.model,
                        created: created,
                        text: text
                    ) {
                        emitSSE(contentChunk.formatted)
                    }

                case .info(let promptTokens, let completionTokens, _):
                    usageAccumulator.setPromptTokens(promptTokens)
                    usageAccumulator.setCompletionTokens(completionTokens)

                case .error(let errorMessage):
                    log.error("[\(requestId)] Generation error: \(errorMessage)")
                    send.send(.inferenceError(requestId: requestId, error: errorMessage, statusCode: 500))
                    return
                }
            }

            // Generation complete
            let usage = usageAccumulator.snapshot
            if let finishChunk = try? formatter.finishChunk(
                id: responseId,
                model: chatRequest.model,
                created: created,
                reason: .stop,
                usage: usage
            ) {
                emitSSE(finishChunk.formatted)
                emitSSE(SSEChunk.done.formatted)
            }

            // Update stats
            providerStats.incrementRequestsServed()
            providerStats.addTokensGenerated(UInt64(usage.completionTokens))

            // Update state
            let cap = await sched.backendCapacity()
            providerState.backendCapacity = cap
            if await sched.capacity().activeRequests == 0 {
                providerState.inferenceActive = false
            }

            // Send completion
            let attestation = computeResponseAttestation(
                identity: signingIdentity,
                requestId: requestId,
                completionTokens: UInt64(max(usage.completionTokens, 0)),
                responseBody: fullResponseText
            )
            send.send(.inferenceComplete(
                requestId: requestId,
                usage: usage.protocolUsageInfo,
                seSignature: attestation.signature,
                responseHash: attestation.hash
            ))

            log.info("[\(requestId)] Complete: \(usage.promptTokens) prompt + \(usage.completionTokens) completion tokens")
        }

        inflightTasks[requestId] = task
    }

    // MARK: - Model Loading

    private func ensureModelLoaded(modelId: String) async throws {
        // Check if already loaded
        if state.currentModel == modelId {
            return
        }

        // Resolve local path
        guard let modelPath = ModelScanner.resolveLocalPath(modelID: modelId) else {
            throw InferenceError.invalidModelDirectory(
                "Model '\(modelId)' not found in local HuggingFace cache"
            )
        }

        logger.info("Loading model: \(modelId) from \(modelPath.path)")

        let container = try await loadModelContainer(from: modelPath)
        await scheduler.loadModel(container: container, modelId: modelId)

        // Update shared state
        state.currentModel = modelId
        state.warmModels = [modelId]
        state.currentModelHash = loopConfig.modelHashes[modelId]

        logger.info("Model loaded: \(modelId)")
    }

    private func loadModelContainer(from directory: URL) async throws -> MLXLMCommon.ModelContainer {
        try await LLMModelFactory.shared.loadContainer(
            from: directory,
            using: LocalTokenizerLoader()
        )
    }

    // MARK: - Cancellation

    private func handleCancellation(requestId: String) async {
        logger.info("Cancelling request: \(requestId)")

        // Cancel in the registry (triggers the token)
        await cancellationRegistry.cancel(requestId: requestId)

        // Cancel in the scheduler
        await scheduler.cancel(requestId: requestId)

        // Cancel the inflight task
        if let task = inflightTasks.removeValue(forKey: requestId) {
            task.cancel()
        }
    }

    private func cancelAllInflight() async {
        let requestIds = Array(inflightTasks.keys)
        for requestId in requestIds {
            await handleCancellation(requestId: requestId)
        }
        inflightTasks.removeAll()
    }

    private func removeInflightTask(requestId: String) {
        inflightTasks.removeValue(forKey: requestId)
    }

    // MARK: - Attestation Challenge

    private func handleAttestationChallenge(
        nonce: String,
        timestamp: String,
        send: SendHandle
    ) async {
        logger.info("Handling attestation challenge (timestamp: \(timestamp))")

        guard let builder = attestationBuilder else {
            logger.warning("No Secure Enclave identity -- cannot respond to attestation challenge")
            return
        }

        do {
            let activeModelHash = state.currentModel.flatMap { modelId in
                loopConfig.modelHashes[modelId]
            }

            let response = try builder.buildChallengeResponse(
                nonce: nonce,
                timestamp: timestamp,
                providerPublicKey: keyPair.publicKeyBase64,
                binaryHash: binaryHash,
                activeModelHash: activeModelHash,
                runtimeHashes: loopConfig.runtimeHashes,
                modelHashes: loopConfig.modelHashes
            )

            send.send(.attestationResponse(AttestationResponsePayload(
                nonce: response.nonce,
                signature: response.signature,
                statusSignature: response.statusSignature,
                publicKey: response.publicKey,
                hypervisorActive: response.hypervisorActive,
                rdmaDisabled: response.rdmaDisabled,
                sipEnabled: response.sipEnabled,
                secureBootEnabled: response.secureBootEnabled,
                binaryHash: response.binaryHash,
                activeModelHash: response.activeModelHash,
                pythonHash: response.pythonHash,
                runtimeHash: response.runtimeHash,
                templateHashes: response.templateHashes,
                modelHashes: response.modelHashes
            )))

            logger.info("Attestation challenge response sent")
        } catch {
            logger.error("Failed to sign attestation challenge: \(error)")
        }
    }

    // MARK: - Helpers

}

// MARK: - Logger wrapper

/// Unified logger that uses os.Logger on macOS.
private struct ProviderLogger: Sendable {
    #if canImport(os)
    private let osLogger: os.Logger
    #endif
    private let category: String

    init(subsystem: String, category: String) {
        self.category = category
        #if canImport(os)
        self.osLogger = os.Logger(subsystem: subsystem, category: category)
        #endif
    }

    func info(_ message: String) {
        #if canImport(os)
        osLogger.info("\(message, privacy: .public)")
        #else
        print("[\(category)] INFO: \(message)")
        #endif
    }

    func warning(_ message: String) {
        #if canImport(os)
        osLogger.warning("\(message, privacy: .public)")
        #else
        print("[\(category)] WARN: \(message)")
        #endif
    }

    func error(_ message: String) {
        #if canImport(os)
        osLogger.error("\(message, privacy: .public)")
        #else
        print("[\(category)] ERROR: \(message)")
        #endif
    }
}

// MARK: - Import bridge

import MLX
import MLXLLM
import MLXLMCommon
