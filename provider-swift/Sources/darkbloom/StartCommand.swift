import Foundation
import ArgumentParser
import ProviderCore

struct Start: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start the provider as a background service.",
        discussion: """
        Scans local MLX models, lets you pick which to serve, then launches
        a launchd background service. Use --model to skip the interactive picker.
        """
    )

    @OptionGroup var configOptions: ConfigOptions

    @Option(help: "Override coordinator WebSocket URL.")
    var coordinatorURL: String?

    @Option(help: "Model ID to serve (repeatable, skips interactive picker).")
    var model: [String] = []

    @Flag(help: "Serve all local models (skips interactive picker).")
    var all = false

    @Option(help: "Idle timeout in minutes before unloading the model.")
    var idleTimeout: UInt64?

    @Flag(inversion: .prefixedNo, help: .hidden)
    var foreground = false

    mutating func run() async throws {
        let snapshot = try loadRuntimeSnapshot(configOptions: configOptions)
        let effectiveCoordinator = coordinatorURL ?? snapshot.config.coordinator.url
        var effectiveConfig = snapshot.config
        if let idleTimeout {
            effectiveConfig.backend.idleTimeoutMins = idleTimeout
        }

        guard let hardware = snapshot.hardware else {
            printError("Cannot start: hardware detection failed (\(snapshot.hardwareError?.localizedDescription ?? "unknown"))")
            throw ExitCode.failure
        }

        guard !snapshot.models.isEmpty else {
            printError("No local MLX models found. Download models to ~/.cache/huggingface/hub/")
            throw ExitCode.failure
        }

        if foreground {
            try await runForeground(
                snapshot: snapshot,
                hardware: hardware,
                config: effectiveConfig,
                coordinatorURL: effectiveCoordinator
            )
        } else {
            try launchDaemon(
                snapshot: snapshot,
                config: effectiveConfig,
                coordinatorURL: effectiveCoordinator
            )
        }
    }

    // MARK: - Foreground (invoked by launchd)

    private func runForeground(
        snapshot: RuntimeSnapshot,
        hardware: HardwareInfo,
        config: ProviderConfig,
        coordinatorURL: String
    ) async throws {
        let selectedModels: [ModelInfo]
        if !model.isEmpty {
            selectedModels = advertisedModels(from: snapshot.models, config: config, modelOverrides: model)
        } else if all {
            selectedModels = snapshot.models
        } else {
            selectedModels = advertisedModels(from: snapshot.models, config: config)
        }

        guard !selectedModels.isEmpty else {
            printError("No models selected.")
            throw ExitCode.failure
        }

        let (models, modelHashes) = attachWeightHashes(to: selectedModels)
        let runtimeHashes = (try? RuntimeHashReporter().report().coordinatorRuntimeHashes)
        let authToken = AuthTokenStore.load()

        print("darkbloom \(ProviderCore.version)")
        print("Backend: mlx-swift")
        print("Config: \(describeConfigPath(snapshot))")
        print("Coordinator: \(coordinatorURL)")
        print("Advertised models: \(models.count)")
        for m in models {
            print("  \(m.id) (\(String(format: "%.1f", m.estimatedMemoryGb)) GB)")
        }

        let loopConfig = ProviderLoopConfig(
            coordinatorURL: coordinatorURL,
            hardware: hardware,
            models: models,
            config: config,
            authToken: authToken,
            runtimeHashes: runtimeHashes,
            modelHashes: modelHashes
        )

        let loop = try ProviderLoop(config: loopConfig)
        try await loop.run()
    }

    // MARK: - Daemon (interactive picker → launchd)

    private func launchDaemon(
        snapshot: RuntimeSnapshot,
        config: ProviderConfig,
        coordinatorURL: String
    ) throws {
        let selectedModelIDs: [String]

        if !model.isEmpty {
            selectedModelIDs = model
        } else if all {
            selectedModelIDs = snapshot.models.map(\.id)
        } else {
            selectedModelIDs = try interactiveModelPicker(snapshot: snapshot, config: config)
        }

        guard !selectedModelIDs.isEmpty else {
            printError("No models selected.")
            throw ExitCode.failure
        }

        try LaunchAgent.installAndStart(
            coordinatorURL: coordinatorURL,
            models: selectedModelIDs,
            idleTimeout: idleTimeout ?? (config.backend.idleTimeoutMins > 0 ? config.backend.idleTimeoutMins : nil)
        )

        let logPath = LaunchAgent.logPath().path
        print("Provider started as background service.")
        print("  Models:  \(selectedModelIDs.count)")
        for id in selectedModelIDs {
            print("    \(id)")
        }
        print("  Logs:    \(logPath)")
        print()
        print("  darkbloom stop    Stop the provider")
        print("  darkbloom status  Check status")
    }

    // MARK: - Interactive Picker

    private func interactiveModelPicker(
        snapshot: RuntimeSnapshot,
        config: ProviderConfig
    ) throws -> [String] {
        let models = snapshot.models.sorted { $0.id < $1.id }

        print()
        print("  Available models:")
        print()
        for (i, m) in models.enumerated() {
            let sizeStr = String(format: "%.1f GB", m.estimatedMemoryGb)
            let quant = m.quantization ?? ""
            print("    [\(i + 1)] \(m.id)  (\(sizeStr)\(quant.isEmpty ? "" : ", \(quant)"))")
        }
        print()
        print("  Select models (comma-separated numbers, or 'all'): ", terminator: "")

        guard let input = readLine()?.trimmingCharacters(in: .whitespaces), !input.isEmpty else {
            return []
        }

        if input.lowercased() == "all" {
            return models.map(\.id)
        }

        let indices = input.split(separator: ",").compactMap { token -> Int? in
            guard let n = Int(token.trimmingCharacters(in: .whitespaces)) else { return nil }
            return n
        }

        var selected: [String] = []
        for idx in indices {
            guard idx >= 1, idx <= models.count else {
                printError("Invalid selection: \(idx) (must be 1-\(models.count))")
                throw ExitCode.failure
            }
            selected.append(models[idx - 1].id)
        }
        return selected
    }
}
