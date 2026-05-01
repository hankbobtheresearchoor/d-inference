import Foundation
import ArgumentParser
import ProviderCore

struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List locally cached MLX models."
    )

    @OptionGroup var configOptions: ConfigOptions

    @Flag(help: "Emit JSON instead of a table.")
    var json = false

    @Flag(help: "Show every discovered local model, ignoring the config enabled_models filter.")
    var all = false

    @Option(help: "Compute an on-demand integrity hash for one model ID.")
    var hash: String?

    mutating func run() async throws {
        if let hash {
            let digest = WeightHasher.computeHash(for: hash)
            guard let digest else {
                throw ValidationError("could not compute weight hash for '\(hash)'")
            }
            if json {
                let payload = HashOutput(model: hash, weightHash: digest)
                try printJSON(payload)
            } else {
                print("\(hash) \(digest)")
            }
            return
        }

        let snapshot = try loadRuntimeSnapshot(configOptions: configOptions)
        let models = advertisedModels(from: snapshot.models, config: snapshot.config, includeDisabled: all)

        if json {
            let payload = ModelsOutput(
                cacheDirectory: ModelScanner.defaultCacheDirectory()?.path,
                filteredByConfig: !all && !snapshot.config.backend.enabledModels.isEmpty,
                models: models
            )
            try printJSON(payload)
            return
        }

        guard !models.isEmpty else {
            print("No local MLX models found.")
            if let cache = ModelScanner.defaultCacheDirectory() {
                print("Cache: \(cache.path)")
            }
            return
        }

        print("Local MLX models")
        printModelTable(models)
    }
}
