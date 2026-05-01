// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "DarkbloomProvider",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ProviderCore", targets: ["ProviderCore"]),
        .executable(name: "darkbloom", targets: ["darkbloom"]),
    ],
    dependencies: [
        .package(path: "../libs/mlx-swift"),
        .package(path: "../libs/mlx-swift-lm"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
        .package(url: "https://github.com/jedisct1/swift-sodium.git", from: "0.9.1"),
        .package(url: "https://github.com/LebJe/TOMLKit.git", from: "0.6.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", exact: "2.22.0"),
    ],
    targets: [
        .target(
            name: "ProviderCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Sodium", package: "swift-sodium"),
                .product(name: "TOMLKit", package: "TOMLKit"),
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            path: "Sources/ProviderCore"
        ),
        .executableTarget(
            name: "darkbloom",
            dependencies: [
                "ProviderCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/darkbloom"
        ),
        .testTarget(
            name: "ProviderCoreTests",
            dependencies: [
                "ProviderCore",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ],
            path: "Tests/ProviderCoreTests"
        ),
    ]
)
