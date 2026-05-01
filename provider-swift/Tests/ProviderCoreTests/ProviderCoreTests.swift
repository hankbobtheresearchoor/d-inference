import Testing
@testable import ProviderCore

@Test func versionExists() {
    #expect(ProviderCore.version.contains("swift"))
}
