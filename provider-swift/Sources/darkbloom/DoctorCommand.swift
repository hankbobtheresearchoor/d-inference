import Foundation
import ArgumentParser
import ProviderCore

struct Doctor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run local provider diagnostics.",
        discussion: "Diagnostics are read-only except for subprocesses used by public ProviderCore checks."
    )

    @OptionGroup var configOptions: ConfigOptions

    @Flag(help: "Treat warning-level checks as failures.")
    var strict = false

    mutating func run() async throws {
        let snapshot = try loadRuntimeSnapshot(configOptions: configOptions)
        let checks = buildDoctorChecks(snapshot: snapshot)

        print("darkbloom doctor")
        print("Config: \(describeConfigPath(snapshot))")
        for check in checks {
            print("\(check.status.marker) \(check.name): \(check.detail)")
        }

        let hasFailure = checks.contains { $0.status == .fail }
        let hasWarning = checks.contains { $0.status == .warn }

        if hasFailure || (strict && hasWarning) {
            throw ExitCode.failure
        }
    }
}

// MARK: - Doctor

enum CheckStatus: Equatable {
    case pass
    case warn
    case fail

    var marker: String {
        switch self {
        case .pass: return "[PASS]"
        case .warn: return "[WARN]"
        case .fail: return "[FAIL]"
        }
    }
}

struct DoctorCheck {
    let name: String
    let status: CheckStatus
    let detail: String
}

func buildDoctorChecks(snapshot: RuntimeSnapshot) -> [DoctorCheck] {
    var checks: [DoctorCheck] = []

    if let hardware = snapshot.hardware {
        checks.append(.init(
            name: "hardware",
            status: .pass,
            detail: "\(hardware.chipName), \(hardware.memoryGb) GB RAM, \(hardware.gpuCores) GPU cores"
        ))
    } else {
        checks.append(.init(
            name: "hardware",
            status: .fail,
            detail: snapshot.hardwareError?.localizedDescription ?? "hardware detection failed"
        ))
    }

    checks.append(.init(
        name: "config",
        status: snapshot.configFileExists ? .pass : .warn,
        detail: snapshot.configFileExists ? "loaded" : "missing, defaults are in memory only"
    ))

    if let cacheDir = ModelScanner.defaultCacheDirectory(),
       FileManager.default.fileExists(atPath: cacheDir.path) {
        checks.append(.init(
            name: "huggingface cache",
            status: .pass,
            detail: cacheDir.path
        ))
    } else {
        checks.append(.init(
            name: "huggingface cache",
            status: .warn,
            detail: "not found"
        ))
    }

    checks.append(.init(
        name: "local mlx models",
        status: snapshot.models.isEmpty ? .warn : .pass,
        detail: "\(snapshot.models.count) discovered"
    ))

    let sipEnabled = checkSIPEnabled()
    checks.append(.init(
        name: "sip",
        status: sipEnabled ? .pass : .fail,
        detail: sipEnabled ? "enabled" : "disabled"
    ))

    let rdmaDisabled = checkRDMADisabled()
    checks.append(.init(
        name: "rdma",
        status: rdmaDisabled ? .pass : .warn,
        detail: rdmaDisabled ? "disabled" : "enabled; allowed for RDMA-aware runtimes"
    ))

    let secureBoot = checkSecureBootEnabled()
    checks.append(.init(
        name: "secure boot",
        status: secureBoot ? .pass : .warn,
        detail: secureBoot ? "enabled" : "not confirmed"
    ))

    let authenticatedRoot = checkAuthenticatedRootEnabled()
    checks.append(.init(
        name: "authenticated root",
        status: authenticatedRoot ? .pass : .warn,
        detail: authenticatedRoot ? "enabled" : "not confirmed"
    ))

    let hardenedRuntime = checkHardenedRuntimeEnabled()
    checks.append(.init(
        name: "hardened runtime",
        status: hardenedRuntime ? .pass : .warn,
        detail: hardenedRuntime ? "enabled" : "not confirmed for this executable"
    ))

    let debuggerAttached = checkDebuggerAttached()
    checks.append(.init(
        name: "debugger",
        status: debuggerAttached ? .fail : .pass,
        detail: debuggerAttached ? "attached" : "not attached"
    ))

    if let binaryHash = selfBinaryHash() {
        checks.append(.init(
            name: "binary hash",
            status: .pass,
            detail: binaryHash
        ))
    } else {
        checks.append(.init(
            name: "binary hash",
            status: .warn,
            detail: "could not compute"
        ))
    }

    return checks
}
