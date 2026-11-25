// Registering the QNN EP plugin library with the Runtime, unregistering it, and
// the module check that follows registration.
//
// QnnEpRegistration needs the bundled plugin but no NPU (label `ort`; skipped in
// builds without the QNN EP). QnnEpNpu needs the NPU itself (label `npu`).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>

#include "edgebench/bundle_modules.hpp"
#include "edgebench/ep_library.hpp"
#include "edgebench/qnn_ep.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"

namespace fs = std::filesystem;
namespace eb = edgebench;
using eb::test::shared_runtime;
using eb::test::TempDir;

namespace {

const fs::path kSystemNonOrtDll = LR"(C:\Windows\System32\version.dll)";

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool any_qnn_device(eb::Runtime& runtime) {
    const std::vector<eb::EpDeviceInfo> devices = runtime.ep_devices();
    return std::any_of(devices.begin(), devices.end(),
                       [](const eb::EpDeviceInfo& d) { return d.ep_name == eb::kQnnEpName; });
}

// Every test starts and ends with nothing registered on the shared runtime.
class QnnEpRegistration : public ::testing::Test {
protected:
    void SetUp() override {
        if (!eb::qnn_ep_in_build())
            GTEST_SKIP() << "this build does not include the QNN execution provider";
        ASSERT_EQ(shared_runtime().ep_library(eb::kQnnEpName), nullptr);
    }
    void TearDown() override {
        const eb::Result<void> r = shared_runtime().unregister_ep_library(eb::kQnnEpName);
        EXPECT_TRUE(r.ok()) << r.error().message;
    }
};

// A copy of a system DLL under a QNN-like name, loaded from outside the bundle.
class ForeignQnnModule {
public:
    ForeignQnnModule() : dir_(L"edgebench_foreign_qnn") {
        path_ = dir_.path() / "QnnForeignTest.dll";
        fs::copy_file(kSystemNonOrtDll, path_);
        module_ = LoadLibraryExW(path_.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    ~ForeignQnnModule() {
        if (module_)
            FreeLibrary(module_);
    }
    ForeignQnnModule(const ForeignQnnModule&) = delete;
    ForeignQnnModule& operator=(const ForeignQnnModule&) = delete;
    bool loaded() const { return module_ != nullptr; }
    const fs::path& path() const { return path_; }

private:
    TempDir dir_;  // destroyed after the module is freed
    fs::path path_;
    HMODULE module_ = nullptr;
};

}  // namespace

TEST_F(QnnEpRegistration, RegistersFromTheBundleByAbsolutePathAndUnregisters) {
    eb::Runtime& runtime = shared_runtime();
    const fs::path expected = eb::qnn_ep_library_path(runtime.bundle_dir());

    eb::Result<eb::EpLibrary*> registered = eb::register_qnn_ep(runtime);
    ASSERT_TRUE(registered.ok()) << registered.error().message;
    eb::EpLibrary* library = registered.value();
    EXPECT_EQ(runtime.ep_library(eb::kQnnEpName), library);
    EXPECT_TRUE(library->registered());
    EXPECT_TRUE(library->path().is_absolute());
    EXPECT_EQ(library->path(), expected);
    EXPECT_EQ(library->registration_name(), eb::kQnnEpName);

    // The plugin is loaded now, from the bundle, and nothing is outside it.
    const eb::BundleModuleCheck check = eb::check_bundle_modules(runtime.bundle_dir());
    EXPECT_TRUE(check.ok()) << check.result().error().message;
    EXPECT_TRUE(std::any_of(check.modules.begin(), check.modules.end(), [&](const auto& m) {
        return eb::is_same_path(eb::resolve_path(m.path).native(),
                                eb::resolve_path(expected).native());
    }));

    // Registering again returns the same registration.
    eb::Result<eb::EpLibrary*> again = eb::register_qnn_ep(runtime);
    ASSERT_TRUE(again.ok()) << again.error().message;
    EXPECT_EQ(again.value(), library);

    const eb::Result<void> unregistered = runtime.unregister_ep_library(eb::kQnnEpName);
    ASSERT_TRUE(unregistered.ok()) << unregistered.error().message;
    EXPECT_EQ(runtime.ep_library(eb::kQnnEpName), nullptr);
    EXPECT_FALSE(any_qnn_device(runtime));

    // And it can be registered again in the same environment.
    eb::Result<eb::EpLibrary*> re_registered = eb::register_qnn_ep(runtime);
    ASSERT_TRUE(re_registered.ok()) << re_registered.error().message;
}

TEST_F(QnnEpRegistration, SameNameFromAnotherPathIsRefused) {
    eb::Runtime& runtime = shared_runtime();
    ASSERT_TRUE(eb::register_qnn_ep(runtime).ok());
    TempDir dir;
    const eb::Result<eb::EpLibrary*> other = runtime.register_ep_library(
        std::string(eb::kQnnEpName), dir.path() / "onnxruntime_providers_qnn.dll");
    ASSERT_FALSE(other.ok());
    EXPECT_EQ(other.error().kind, "ep_library_conflict");
    EXPECT_TRUE(runtime.ep_library(eb::kQnnEpName)->registered());
}

TEST_F(QnnEpRegistration, ALeaseBlocksUnregistration) {
    eb::Runtime& runtime = shared_runtime();
    eb::Result<eb::EpLibrary*> registered = eb::register_qnn_ep(runtime);
    ASSERT_TRUE(registered.ok()) << registered.error().message;
    eb::EpLibrary* library = registered.value();
    {
        eb::EpLibrary::Lease lease = library->lease();
        EXPECT_TRUE(lease.held());
        EXPECT_EQ(library->active_leases(), 1);
        const eb::Result<void> refused = runtime.unregister_ep_library(eb::kQnnEpName);
        ASSERT_FALSE(refused.ok());
        EXPECT_EQ(refused.error().kind, "ep_library_in_use");
        EXPECT_TRUE(contains(refused.error().message, "1 session(s)")) << refused.error().message;
        EXPECT_EQ(runtime.ep_library(eb::kQnnEpName), library);
        EXPECT_TRUE(library->registered());

        eb::EpLibrary::Lease moved = std::move(lease);
        EXPECT_FALSE(lease.held());
        EXPECT_EQ(library->active_leases(), 1);
    }
    EXPECT_EQ(library->active_leases(), 0);
    const eb::Result<void> r = runtime.unregister_ep_library(eb::kQnnEpName);
    EXPECT_TRUE(r.ok()) << r.error().message;
}

TEST_F(QnnEpRegistration, AForeignQnnModuleFailsTheRegistration) {
    eb::Runtime& runtime = shared_runtime();
    ForeignQnnModule foreign;
    ASSERT_TRUE(foreign.loaded());
    const eb::Result<eb::EpLibrary*> registered = eb::register_qnn_ep(runtime);
    ASSERT_FALSE(registered.ok());
    const eb::Error& e = registered.error();
    EXPECT_EQ(e.exit_code, eb::ExitCode::ort_load_failure);
    EXPECT_EQ(e.kind, "module_outside_bundle");
    EXPECT_TRUE(contains(e.message, "QnnForeignTest.dll (" +
                                        eb::path_to_utf8(eb::resolve_path(foreign.path())) + ")"))
        << e.message;
    // Rolled back: nothing stays registered after a failed check.
    EXPECT_EQ(runtime.ep_library(eb::kQnnEpName), nullptr);
    EXPECT_FALSE(any_qnn_device(runtime));
}

// Not specific to the QNN EP, but needs ONNX Runtime and runs in every build.
TEST(EpLibraryRegistration, MissingLibraryIsALoadFailure) {
    eb::Runtime& runtime = shared_runtime();
    TempDir dir;
    const eb::Result<eb::EpLibrary*> r =
        runtime.register_ep_library("EdgeBenchTestEp", dir.path() / "missing_ep.dll");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::ort_load_failure);
    EXPECT_EQ(r.error().kind, "ep_library_not_found");
    EXPECT_EQ(runtime.ep_library("EdgeBenchTestEp"), nullptr);
}

TEST(EpLibraryRegistration, ADllThatIsNotAnEpLibraryIsALoadFailure) {
    eb::Runtime& runtime = shared_runtime();
    TempDir dir;
    const fs::path fake = dir.path() / "not_an_ep_library.dll";
    fs::copy_file(kSystemNonOrtDll, fake);
    const eb::Result<eb::EpLibrary*> r = runtime.register_ep_library("EdgeBenchTestEp", fake);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().exit_code, eb::ExitCode::ort_load_failure);
    EXPECT_EQ(r.error().kind, "ep_registration_failed");
    EXPECT_TRUE(r.error().ort_error_name.has_value());
    EXPECT_TRUE(contains(r.error().message, eb::path_to_utf8(fake))) << r.error().message;
    EXPECT_EQ(runtime.ep_library("EdgeBenchTestEp"), nullptr);
}

TEST(EpLibraryRegistration, RelativePathsAreRefused) {
    const eb::Result<std::unique_ptr<eb::EpLibrary>> r = eb::EpLibrary::register_library(
        shared_runtime().env(), "EdgeBenchTestEp", "onnxruntime_providers_qnn.dll");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().kind, "ep_library_path_not_absolute");
}

// --------------------------------------------------------------------------- NPU

TEST(QnnEpNpu, RegistersFindsExactlyOneNpuAndUnregisters) {
    const eb::test::QnnNpuProbe probe = eb::test::probe_qnn_npu();
    if (const std::optional<std::string> reason = eb::test::npu_skip_reason(probe))
        GTEST_SKIP() << *reason;
    ASSERT_FALSE(probe.registration_error) << probe.registration_error->message;

    eb::Runtime& runtime = shared_runtime();
    eb::Result<eb::EpLibrary*> registered = eb::register_qnn_ep(runtime);
    ASSERT_TRUE(registered.ok()) << registered.error().message;

    // Modules checked again after registration: the plugin is in the bundle.
    const eb::BundleModuleCheck check = eb::check_bundle_modules(runtime.bundle_dir());
    EXPECT_TRUE(check.ok()) << check.result().error().message;

    const std::vector<eb::EpDeviceInfo> devices = runtime.ep_devices();
    const eb::NpuSelection selection = eb::select_qnn_npu(devices);
    ASSERT_TRUE(selection.ok()) << selection.message;
    EXPECT_EQ(selection.candidate_count, 1);
    const eb::EpDeviceInfo& npu = devices[*selection.index];
    EXPECT_EQ(npu.ep_name, eb::kQnnEpName);
    EXPECT_EQ(npu.device_type, "NPU");
    std::printf("selected: %s\nall devices: %s\n", eb::describe_device(npu).c_str(),
                eb::describe_devices(devices).c_str());

    // The handle a session will attach to is that same device.
    eb::Result<Ort::ConstEpDevice> handle =
        eb::ep_device_handle(runtime.env(), *selection.index, npu);
    ASSERT_TRUE(handle.ok()) << handle.error().message;
    EXPECT_STREQ(handle.value().EpName(), "QNNExecutionProvider");
    EXPECT_EQ(handle.value().Device().Type(), OrtHardwareDeviceType_NPU);
    EXPECT_EQ(handle.value().Device().DeviceId(), npu.device_id);

    // A stale expectation is detected instead of attaching to the wrong device.
    eb::EpDeviceInfo other = npu;
    other.device_id ^= 1;
    eb::Result<Ort::ConstEpDevice> stale =
        eb::ep_device_handle(runtime.env(), *selection.index, other);
    ASSERT_FALSE(stale.ok());
    EXPECT_EQ(stale.error().kind, "ep_devices_changed");

    const eb::Result<void> unregistered = runtime.unregister_ep_library(eb::kQnnEpName);
    ASSERT_TRUE(unregistered.ok()) << unregistered.error().message;
    EXPECT_FALSE(any_qnn_device(runtime));
    EXPECT_FALSE(eb::select_qnn_npu(runtime.ep_devices()).ok());
}
