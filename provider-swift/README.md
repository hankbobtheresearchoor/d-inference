# Swift Provider Cutover Notes

This package is the Swift-native provider candidate. `Package.swift` currently
defines the `ProviderCore` library and `darkbloom` executable, requires
Swift tools 6.1, targets macOS 15, and depends on local `../libs/mlx-swift`
and `../libs/mlx-swift-lm`.

## Local Gates

Run these before any release-script cutover:

```bash
swift test --package-path provider-swift
swift build -c release --package-path provider-swift
provider-swift/.build/release/darkbloom --help
```

The CLI is not release-ready while `darkbloom serve` remains a stub. At the
time of this pass, `Sources/darkbloom/main.swift` prints `Backend: mlx-swift`
but exits failure for serving because coordinator event handling and inference
dispatch are not wired into the CLI yet.

## Compatibility Decisions To Lock

- Backend identifier: use `mlx-swift` for the Swift provider cutover. Current
  repo state is split: the Swift CLI prints `mlx-swift`, existing Swift tests
  still use `mlx_swift_lm`, and the coordinator private-text gate checks for
  `inprocess-mlx` in `coordinator/internal/registry/registry.go`.
- Runtime hashes: Swift releases should omit `python_hash` and `runtime_hash`.
  Keep `binary_hash`, `bundle_hash`, and any template/model hashes that are
  still meaningful.
- Runtime verification: the coordinator must become backend-aware. Existing
  `SyncRuntimeManifest` and `verifyRuntimeHashes` logic in
  `coordinator/internal/api/server.go` applies Python/runtime hash requirements
  globally, so a Swift provider with no Python hashes will fail whenever old
  active releases still contribute Python/runtime hashes. If all old releases
  are deactivated, the manifest becomes nil and providers lose private-text
  eligibility because registration marks `RuntimeManifestChecked=false`.

## Cutover Checklist

1. Finish the Swift runtime path.
   - `darkbloom serve` must connect to `/ws/provider`, register, handle
     inference, cancellation, reconnects, and attestation challenges.
   - Registration should send `backend: "mlx-swift"`, encrypted response
     chunks, privacy capabilities, binary/model/template hashes as applicable,
     and no `python_hash` or `runtime_hash`.
   - Update Swift tests that still use `mlx_swift_lm` after the coordinator
     accepts the canonical backend string.

2. Update `scripts/build-bundle.sh`.
   - Replace the Rust provider build with
     `swift build -c release --package-path provider-swift`.
   - Set the provider binary path to
     `provider-swift/.build/release/darkbloom`.
   - Remove portable Python setup, `vllm-mlx` installs, import checks, PyO3
     build environment, `install_name_tool` libpython rewrites, and Python
     runtime hash generation.
   - Bundle `bin/darkbloom`; keep `bin/eigeninference-enclave` only if Phase 5
     has not fully merged Secure Enclave support into the Swift provider.
   - Keep signing with `scripts/entitlements.plist`, then compute
     `binary_hash` and `bundle_hash` after signing/stapling.
   - Update app bundle minimum OS from 14.0 to 15.0 if this script still builds
     the app bundle.

3. Update `.github/workflows/release.yml`.
   - Remove Cargo/PyO3/PBS/uv/vllm-mlx setup and cache entries.
   - Add SwiftPM cache coverage for `provider-swift/.build` and
     `provider-swift/Package.resolved`.
   - Build with `swift build -c release --package-path provider-swift`.
   - Test with `swift test --package-path provider-swift`.
   - Assemble the provider bundle from the Swift `darkbloom` binary without a
     `python/` directory and without libpython rpath patching.
   - Upload only the bundle and DMG unless another Swift-native artifact is
     explicitly introduced.
   - Register releases without `python_hash` and `runtime_hash`; keep
     `binary_hash`, `bundle_hash`, `template_hashes`, `url`, `platform`, and
     `changelog`.
   - Update release notes so they no longer advertise a vllm-mlx runtime hash.

4. Update `scripts/install.sh` and the embedded coordinator copy.
   - Remove the Python runtime verification/download/install step.
   - Remove `PYTHON_BIN`, PBS fallback, site-packages fallback, and vllm-mlx
     import checks.
   - Replace Python-based model catalog parsing with a Swift CLI helper or a
     shell-only parser. Prefer moving selection/download logic into `darkbloom`
     so a fresh macOS install still has no Python prerequisite.
   - If Secure Enclave is merged into the main binary, replace direct
     `eigeninference-enclave info` calls with the new `darkbloom` command.
   - Reduce the displayed step count and summary to match the no-Python flow.

5. Update coordinator release and runtime compatibility.
   - In `coordinator/internal/api/release_handlers.go`, require
     `bundle_hash` because `install.sh` verifies it.
   - In `coordinator/internal/store/interface.go`, update `Release` comments
     and consider adding a `backend` or `runtime_kind` field if Rust and Swift
     releases will overlap.
   - In `coordinator/internal/api/server.go`, make runtime verification
     backend-aware: legacy providers should keep Python/runtime hash checks,
     while `mlx-swift` providers should pass without those fields and rely on
     signed binary hash, template hashes, model hashes, and attestation status.
   - In `coordinator/internal/registry/registry.go`, update
     `providerSupportsPrivateTextLocked` to accept the canonical Swift backend
     and to stop depending on Python-specific capability wording once the
     replacement runtime policy is in place.
   - Update coordinator tests that assert `inprocess-mlx` or old runtime hash
     semantics.

6. Production validation.
   - Publish a dev release first and install from the dev coordinator.
   - Run a side-by-side Rust vs Swift provider comparison on the same model and
     machine: registration, chat streaming, cancellation, reconnect, attested
     binary hash, model weight hash, thermal behavior, and 48-hour soak.
   - Keep Rust release artifacts active until Swift has passed soak and the
     coordinator can route both backends intentionally.
