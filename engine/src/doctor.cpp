#include "edgebench/doctor.hpp"

#include <onnxruntime_cxx_api.h>

#include <windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <system_error>

#include "bundle_pins.hpp"
#include "doctor_probes.hpp"
#include "edgebench/bundle_modules.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/provider.hpp"
#include "edgebench/qnn_backend_info.hpp"
#include "edgebench/qnn_ep.hpp"
#include "edgebench/run_config.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/sha256.hpp"

namespace edgebench {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <std::size_t N>
std::vector<FilePin> to_pins(const std::array<pins::PinnedFile, N>& files) {
    std::vector<FilePin> out;
    for (const pins::PinnedFile& f : files)
        out.push_back({std::string(f.name), f.size_bytes, std::string(f.sha256)});
    return out;
}

template <std::size_t N>
std::span<const std::byte> bytes_of(const unsigned char (&data)[N]) {
    return {reinterpret_cast<const std::byte*>(data), N};
}

BundleFile read_bundle_file(const fs::path& dir, const std::string& name) {
    BundleFile f;
    f.name = name;
    f.path = dir / fs::path(std::u8string(name.begin(), name.end()));
    std::error_code ec;
    f.present = fs::is_regular_file(f.path, ec);
    if (!f.present)
        return f;
    f.path = resolve_path(f.path);
    if (const std::uintmax_t size = fs::file_size(f.path, ec); !ec)
        f.size_bytes = size;
    if (Result<std::string> sha = sha256_file(f.path))
        f.sha256 = sha.value();
    if (const std::optional<ModuleFileVersion> v = read_module_file_version(f.path)) {
        f.product_version = v->product_version;
        f.file_version = v->file_version;
    }
    return f;
}

std::vector<BundleFile> read_bundle_files(const fs::path& dir, const std::vector<FilePin>& pins) {
    std::vector<BundleFile> out;
    for (const FilePin& p : pins)
        out.push_back(read_bundle_file(dir, p.name));
    return out;
}

Result<void> write_file(const fs::path& path, std::span<const std::byte> data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!out) {
        return Error{ExitCode::output_error, "write_failed", "cannot write " + path_to_utf8(path)};
    }
    return {};
}

// A directory of this process under %TEMP%, removed at the end.
class TempDirectory {
public:
    TempDirectory() {
        std::error_code ec;
        const fs::path base = fs::temp_directory_path(ec);
        path_ = base / ("edgebench-doctor-" + std::to_string(GetCurrentProcessId()) + "-" +
                        std::to_string(GetTickCount64()));
        ok_ = !ec && fs::create_directories(path_, ec) && !ec;
    }
    ~TempDirectory() { (void)remove(); }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    bool ok() const { return ok_; }
    const fs::path& path() const { return path_; }
    // Removes the directory; the error message if it could not be removed.
    std::optional<std::string> remove() {
        if (!ok_ || removed_)
            return std::nullopt;
        std::error_code ec;
        fs::remove_all(path_, ec);
        removed_ = !ec;
        if (ec)
            return path_to_utf8(path_) + ": " + ec.message();
        return std::nullopt;
    }

private:
    fs::path path_;
    bool ok_ = false;
    bool removed_ = false;
};

RunConfig probe_config(const fs::path& model, bool fp16) {
    RunConfig c;
    c.model = model;
    c.provider = ProviderKind::npu;
    c.npu_fp16 = fp16;
    return c;  // strict, burst, 4 threads, IoBinding, generated input with seed 0: as `run`
}

// One probe: a strict session, one inference, the output compared with the
// expected bytes. `output` receives the output bytes.
ProbeOutcome run_probe(Provider& provider, const RunConfig& config, std::string model_name,
                       std::string compute, std::span<const std::byte> expected,
                       const fs::path& bundle, std::vector<LoadedModule>& outside,
                       std::vector<std::byte>* output) {
    ProbeOutcome o;
    o.model = std::move(model_name);
    o.compute = std::move(compute);
    Result<std::unique_ptr<BenchSession>> session = provider.create_session(config.model, config);
    if (!session) {
        o.error = session.error();
        return o;
    }
    o.session_create_ms = session.value()->session_create_ms();
    BundleModuleCheck modules = check_bundle_modules(bundle);
    outside.insert(outside.end(), modules.outside.begin(), modules.outside.end());
    if (Result<void> run = session.value()->run_once(); !run) {
        o.error = run.error();
        return o;
    }
    const TensorView v = session.value()->output(0);
    compare_probe_output(o, v.bytes, expected);
    if (output)
        output->assign(v.bytes.begin(), v.bytes.end());
    return o;
}

// The NPU part: backend versions (before any session), the cache directory, the
// probes and the context check. Fills the facts of those checks.
struct NpuFacts {
    BackendVersionFacts backend;
    ProbeFacts probe;
    ContextFacts context;
    CurrentCacheEnvironment environment;
    bool environment_known = false;
    std::optional<std::string> qnn_ep_version;
    std::optional<std::string> sdk_version;
    double backend_ms = 0, probe_ms = 0, context_ms = 0;
};

}  // namespace

DoctorReport run_doctor(const DoctorConfig& config) {
    const Clock::time_point start = Clock::now();
    DoctorReport r;
    r.build = build_info();
    r.system = query_system_info();
    r.system.power = query_power_state();
    r.bundle_dir = executable_dir();
    r.versions.onnxruntime_pinned = std::string(pins::kOnnxRuntimeVersion);
    r.versions.qnn_ep_pinned = std::string(pins::kQnnEpVersion);
    r.versions.qairt_sdk_pinned = std::string(pins::kQairtSdkVersion);
    r.versions.qnn_in_build = qnn_ep_in_build();

    // A quiet moment first: the background load sample, before this process works.
    Clock::time_point t = Clock::now();
    if (config.load_sample.count() > 0)
        r.system.load = sample_cpu_load(config.load_sample);
    const double load_ms = ms_since(t);

    // ONNX Runtime: load, the pinned files.
    t = Clock::now();
    const std::vector<FilePin> ort_pins = to_pins(pins::kOnnxRuntimeFiles);
    const std::vector<FilePin> qnn_pins = to_pins(pins::kQnnFiles);
    const OrtLoadResult load = load_onnxruntime(r.bundle_dir);
    if (load.ok())
        r.versions.onnxruntime_loaded = load.ort_version;
    const std::vector<BundleFile> ort_files = read_bundle_files(r.bundle_dir, ort_pins);
    double ort_ms = ms_since(t);

    t = Clock::now();
    const std::vector<BundleFile> qnn_files = read_bundle_files(r.bundle_dir, qnn_pins);
    QnnRegistration registration;
    DeviceFacts devices;
    devices.qnn_in_build = !qnn_pins.empty();
    NpuFacts npu;
    npu.backend.pinned_sdk_version = std::string(pins::kQairtSdkVersion);
    if (const BundleFile* htp = [&]() -> const BundleFile* {
            for (const BundleFile& f : qnn_files)
                if (f.name == "QnnHtp.dll")
                    return &f;
            return nullptr;
        }())
        npu.backend.backend_file_product_version = htp->product_version;
    CacheDirFacts cache;
    std::vector<LoadedModule> modules, outside;
    double qnn_ms = 0, devices_ms = 0, cache_ms = 0;

    const std::string no_npu_runtime = "ONNX Runtime is not loaded";
    std::optional<std::string> npu_unavailable;  // why the NPU checks cannot run
    {
        std::unique_ptr<Runtime> runtime;
        if (load.ok()) {
            Result<std::unique_ptr<Runtime>> created = Runtime::create(load);
            if (created)
                runtime = std::move(created).value();
            else
                devices.enumeration_error = created.error();
        }
        if (!runtime) {
            npu_unavailable = no_npu_runtime;
        } else {
            if (!qnn_pins.empty() || fs::is_regular_file(qnn_ep_library_path(r.bundle_dir))) {
                registration.attempted = true;
                if (Result<EpLibrary*> reg = register_qnn_ep(*runtime); !reg)
                    registration.error = reg.error();
            }
            qnn_ms = ms_since(t);

            t = Clock::now();
            try {
                devices.devices = runtime->ep_devices();
            } catch (const Ort::Exception& e) {
                devices.enumeration_error =
                    ort_error("ep_enumeration_failed", "enumerating EP devices",
                              e.GetOrtErrorCode(), e.what());
            }
            devices.qnn_registered = registration.attempted && !registration.error;
            if (devices.qnn_registered && devices.devices) {
                devices.selection = select_qnn_npu(*devices.devices);
                for (const EpDeviceInfo& d : *devices.devices) {
                    if (d.ep_name == kQnnEpName && d.ep_metadata.contains("version")) {
                        npu.qnn_ep_version = d.ep_metadata.at("version");
                        break;
                    }
                }
            }
            devices_ms = ms_since(t);

            if (qnn_pins.empty())
                npu_unavailable = "this build has no QNN execution provider";
            else if (config.quick)
                npu_unavailable = std::string(kSkippedByQuick);
            else if (!devices.qnn_registered)
                npu_unavailable = "the QNN execution provider is not registered";
            else if (!devices.selection || !devices.selection->ok())
                npu_unavailable = "no NPU to run on";

            // The NPU provider: its creation asks the QNN EP for the backend build
            // id, which must happen before the first QNN session in the process.
            t = Clock::now();
            std::unique_ptr<Provider> int8_provider;
            TempDirectory temp;
            const fs::path int8_model = temp.path() / "npu_probe_int8.onnx";
            const fs::path fp32_model = temp.path() / "npu_probe_fp32.onnx";
            if (!npu_unavailable) {
                if (!temp.ok()) {
                    npu.probe.probes.push_back(
                        {"npu_probe_int8.onnx", "int8",
                         Error{ExitCode::output_error, "temporary_directory_failed",
                               "cannot create a temporary directory under %TEMP%"}});
                    npu_unavailable = "no temporary directory";
                } else {
                    (void)write_file(int8_model, bytes_of(doctor_resources::kNpuProbeInt8Model));
                    (void)write_file(fp32_model, bytes_of(doctor_resources::kNpuProbeFp32Model));
                    Result<std::unique_ptr<Provider>> created =
                        make_provider(*runtime, probe_config(int8_model, false));
                    if (created) {
                        int8_provider = std::move(created).value();
                    } else {
                        npu.probe.probes.push_back(
                            {"npu_probe_int8.onnx", "int8", created.error()});
                    }
                }
            }
            if (int8_provider) {
                const ProviderDescription& d = int8_provider->describe();
                if (d.qnn) {
                    npu.backend.backend_build = d.qnn->backend_build;
                    npu.backend.sdk_version = d.qnn->sdk_version;
                    npu.backend.source = d.qnn->backend_build_source;
                    npu.sdk_version = d.qnn->sdk_version;
                }
                npu.environment.ort_version = load.ort_version;
                npu.environment.ep_version = d.ep_version;
                npu.environment.qnn_sdk_version = d.qnn ? d.qnn->sdk_version : std::nullopt;
                if (d.qnn) {
                    const fs::path backend(
                        std::u8string(d.qnn->backend_path.begin(), d.qnn->backend_path.end()));
                    if (const std::optional<ModuleFileVersion> v =
                            read_module_file_version(backend))
                        npu.environment.qnn_backend_file_version = v->product_version;
                }
                if (d.ep_device) {
                    CacheKeyFields device_only;
                    device_only.device =
                        CacheDevice{d.ep_device->device.device_type, d.ep_device->device.vendor_id,
                                    d.ep_device->device.device_id};
                    npu.environment.device = cache_key_json(device_only).at("device");
                }
                npu.environment_known = true;
            } else {
                npu.backend.not_applicable =
                    npu_unavailable ? *npu_unavailable : "the NPU provider could not be created";
            }
            npu.backend_ms = ms_since(t);

            // The cache directory (no QNN calls).
            t = Clock::now();
            cache.prune_requested = config.prune_cache;
            Result<fs::path> dir = config.ep_cache_dir ? Result<fs::path>(*config.ep_cache_dir)
                                                       : default_ep_cache_dir();
            if (!dir) {
                cache.dir_error = dir.error();
            } else {
                std::error_code ec;
                cache.dir = fs::absolute(dir.value(), ec);
                cache.exists = fs::is_directory(cache.dir, ec);
                cache.environment_known = npu.environment_known;
            }
            if (!cache.exists && !cache.dir_error && !cache.dir.empty()) {
                // A directory that does not exist yet is fine if it can be created:
                // its nearest existing ancestor must be a directory.
                std::error_code ec;
                fs::path ancestor = cache.dir.parent_path();
                while (!ancestor.empty() && !fs::exists(ancestor, ec) &&
                       ancestor != ancestor.parent_path())
                    ancestor = ancestor.parent_path();
                if (fs::exists(cache.dir, ec) || (!ancestor.empty() && fs::exists(ancestor, ec) &&
                                                  !fs::is_directory(ancestor, ec))) {
                    cache.dir_error =
                        Error{ExitCode::output_error, "ep_cache_dir_not_writable",
                              path_to_utf8(cache.dir) + " cannot be created: " +
                                  path_to_utf8(fs::exists(cache.dir, ec) ? cache.dir : ancestor) +
                                  " is not a directory"};
                }
            }
            if (cache.exists) {
                if (Result<fs::path> ready = prepare_ep_cache_dir(cache.dir); ready) {
                    cache.writable = true;
                } else {
                    cache.writable = false;
                    cache.dir_error = ready.error();
                }
                std::error_code ec;
                for (const fs::directory_entry& e : fs::directory_iterator(cache.dir, ec)) {
                    if (!e.is_directory(ec))
                        continue;
                    const std::optional<CacheManifest> m = read_cache_manifest(e.path());
                    if (!m) {
                        cache.foreign.push_back(e.path());
                        continue;
                    }
                    CacheEntryFacts entry;
                    entry.dir = e.path();
                    entry.key_id = m->key_id;
                    entry.context_model = m->context_model;
                    if (const auto it = m->key_fields.find("model_sha256");
                        it != m->key_fields.end() && it->is_string())
                        entry.model_sha256 = it->get<std::string>();
                    for (const CacheFile& f : m->files)
                        entry.size_bytes += f.size_bytes;
                    if (npu.environment_known)
                        entry.stale_fields = stale_key_fields(m->key_fields, npu.environment);
                    if (config.prune_cache && !entry.stale_fields.empty() && !cache.prune_error) {
                        if (Result<std::vector<fs::path>> deleted =
                                delete_cache_entry(cache.dir, entry.dir);
                            deleted)
                            entry.pruned = true;
                        else
                            cache.prune_error = deleted.error();
                    }
                    if (!entry.pruned)
                        cache.total_bytes += entry.size_bytes;
                    cache.entries.push_back(std::move(entry));
                }
            }
            cache_ms = ms_since(t);

            // The probes: the first QNN sessions of the process.
            t = Clock::now();
            std::vector<std::byte> compiled_int8_output;
            if (int8_provider) {
                npu.probe.probes.push_back(run_probe(
                    *int8_provider, probe_config(int8_model, false), "npu_probe_int8.onnx", "int8",
                    bytes_of(doctor_resources::kNpuProbeInt8Expected), r.bundle_dir,
                    npu.probe.outside, &compiled_int8_output));
                const RunConfig fp16_config = probe_config(fp32_model, true);
                Result<std::unique_ptr<Provider>> fp16_provider =
                    make_provider(*runtime, fp16_config);
                if (!fp16_provider) {
                    npu.probe.probes.push_back(
                        {"npu_probe_fp32.onnx", "fp16", fp16_provider.error()});
                } else {
                    npu.probe.probes.push_back(
                        run_probe(*fp16_provider.value(), fp16_config, "npu_probe_fp32.onnx",
                                  "fp16", bytes_of(doctor_resources::kNpuProbeFp32Expected),
                                  r.bundle_dir, npu.probe.outside, nullptr));
                }
            } else if (npu.probe.probes.empty()) {
                npu.probe.not_applicable = npu_unavailable.value_or("no NPU provider");
            }
            npu.probe_ms = ms_since(t);

            // The EP context of the INT8 probe, in the temporary directory only.
            t = Clock::now();
            bool probes_ok =
                int8_provider && !npu.probe.probes.empty() && !npu.probe.probes.front().error;
            if (!probes_ok) {
                npu.context.not_applicable = int8_provider
                                                 ? "the INT8 probe did not run"
                                                 : npu_unavailable.value_or("no NPU provider");
            } else {
                RunConfig ctx_config = probe_config(int8_model, false);
                ctx_config.ep_cache = EpCacheMode::build;
                ctx_config.ep_cache_dir = temp.path();
                Result<std::unique_ptr<Provider>> ctx_provider =
                    make_provider(*runtime, ctx_config);
                const fs::path ctx_dir = temp.path() / "context";
                std::error_code ec;
                fs::create_directories(ctx_dir, ec);
                const fs::path ctx_model = ctx_dir / "npu_probe_int8_ctx.onnx";
                if (!ctx_provider) {
                    npu.context.error = ctx_provider.error();
                } else {
                    Provider& p = *ctx_provider.value();
                    Result<std::unique_ptr<BenchSession>> built =
                        p.create_context_building_session(int8_model, ctx_config, ctx_model);
                    if (!built) {
                        npu.context.error = built.error();
                    } else {
                        npu.context.build_ms = built.value()->session_create_ms();
                        built.value().reset();
                        Result<std::unique_ptr<BenchSession>> loaded =
                            p.create_session(ctx_model, ctx_config);
                        if (!loaded) {
                            Error e = loaded.error();
                            if (e.exit_code != ExitCode::ort_load_failure) {
                                e.exit_code = ExitCode::ort_runtime_error;
                                e.kind = "ep_context_load_failed";
                            }
                            npu.context.error = e;
                        } else {
                            npu.context.load_ms = loaded.value()->session_create_ms();
                            BundleModuleCheck m = check_bundle_modules(r.bundle_dir);
                            npu.probe.outside.insert(npu.probe.outside.end(), m.outside.begin(),
                                                     m.outside.end());
                            if (Result<void> run = loaded.value()->run_once(); !run) {
                                npu.context.error = run.error();
                            } else {
                                const TensorView v = loaded.value()->output(0);
                                npu.context.outputs_equal = std::equal(
                                    v.bytes.begin(), v.bytes.end(), compiled_int8_output.begin(),
                                    compiled_int8_output.end());
                            }
                        }
                        const ContextCompatibility cc = p.context_compatibility(ctx_model);
                        npu.context.compatibility_info = cc.info;
                        npu.context.verdict = cc.verdict;
                        if (!cc.verdict) {
                            npu.context.verdict_note =
                                "the QNN EP cannot rate compatibility info once a QNN session "
                                "has existed in the process, which building the context needs";
                        }
                        if (cc.info) {
                            if (const std::optional<QnnCompatibilityInfo> info =
                                    parse_qnn_compatibility_info(*cc.info))
                                npu.context.htp_arch = htp_arch_name(*info);
                        }
                    }
                }
            }
            npu.context_ms = ms_since(t);
            npu.context.cleanup_error = temp.remove();

            // Every ONNX Runtime / QNN module the checks loaded, while the runtime lives.
            BundleModuleCheck final_check = check_bundle_modules(r.bundle_dir);
            modules = std::move(final_check.modules);
            outside = std::move(final_check.outside);
        }
        if (!load.ok()) {
            BundleModuleCheck c = check_bundle_modules(r.bundle_dir);
            modules = std::move(c.modules);
            outside = std::move(c.outside);
        }
    }  // providers and the runtime are released here, before the report is built

    if (npu_unavailable && !npu.backend.not_applicable && !npu.environment_known)
        npu.backend.not_applicable = *npu_unavailable;
    if (npu_unavailable && npu.probe.probes.empty() && !npu.probe.not_applicable)
        npu.probe.not_applicable = *npu_unavailable;
    if (npu_unavailable && !npu.context.not_applicable && !npu.context.error &&
        !npu.context.outputs_equal)
        npu.context.not_applicable = *npu_unavailable;
    if (!cache.dir_error && cache.dir.empty()) {
        // The runtime never came up: the directory was not examined.
        Result<fs::path> dir =
            config.ep_cache_dir ? Result<fs::path>(*config.ep_cache_dir) : default_ep_cache_dir();
        if (dir) {
            std::error_code ec;
            cache.dir = fs::absolute(dir.value(), ec);
            cache.exists = fs::is_directory(cache.dir, ec);
        } else {
            cache.dir_error = dir.error();
        }
    }
    r.versions.qnn_ep_reported = npu.qnn_ep_version;
    r.versions.qairt_sdk_reported = npu.sdk_version;

    auto add = [&r](DoctorCheck c, double ms) {
        c.duration_ms = ms;
        r.checks.push_back(std::move(c));
    };
    add(check_ort_bundle(load, ort_files, ort_pins, modules, outside, r.system.emulated()), ort_ms);
    add(check_qnn_bundle(qnn_files, qnn_pins, registration), qnn_ms);
    add(check_backend_versions(npu.backend), npu.backend_ms);
    add(check_devices(devices), devices_ms);
    add(check_npu_probe(npu.probe), npu.probe_ms);
    add(check_npu_context(npu.context), npu.context_ms);
    add(check_ep_cache_dir(cache), cache_ms);
    add(check_power(r.system.power), 0);
    add(check_background_load(r.system.load), load_ms);
    r.status = overall_status(r.checks);
    r.exit_code = overall_exit_code(r.checks);
    r.duration_ms = ms_since(start);
    return r;
}

std::string format_text(const DoctorReport& r) {
    std::ostringstream out;
    const SystemInfo& s = r.system;
    out << "edgebench " << version() << " (" << r.build.git_commit.value_or("unknown commit")
        << (r.build.git_dirty.value_or(false) ? ", dirty" : "") << ", " << r.build.build_type
        << ")\n";
    out << "ONNX Runtime " << r.versions.onnxruntime_loaded.value_or("(not loaded)") << " (pinned "
        << r.versions.onnxruntime_pinned << ")";
    if (r.versions.qnn_in_build) {
        out << ", QNN EP " << r.versions.qnn_ep_reported.value_or("(not registered)") << " (pinned "
            << r.versions.qnn_ep_pinned << "), QAIRT SDK "
            << r.versions.qairt_sdk_reported.value_or("(not reported)") << " (release declares "
            << r.versions.qairt_sdk_pinned << ")";
    }
    out << "\n";
    out << s.os.name << " " << s.os.version.value_or("") << " (build " << s.os.build.value_or("?")
        << ", " << to_string(s.os.architecture) << "), " << s.cpu.name.value_or("CPU unknown")
        << ", " << s.device.manufacturer.value_or("?") << " " << s.device.model.value_or("?")
        << "\n";
    out << "bundle " << path_to_utf8(r.bundle_dir) << "\n\n";

    for (const DoctorCheck& c : r.checks) {
        char head[64];
        std::string status(to_string(c.status));
        if (c.status == CheckStatus::not_applicable)
            status = "n/a";
        for (char& ch : status)
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        std::snprintf(head, sizeof(head), "%-5s %-17s ", status.c_str(), c.name.c_str());
        out << head << c.summary << "\n";
        for (const std::string& line : c.lines)
            out << "                        " << line << "\n";
        if (c.status == CheckStatus::warn || c.status == CheckStatus::fail) {
            if (c.code)
                out << "                        [" << *c.code << "]"
                    << (c.status == CheckStatus::fail
                            ? " exit code " + std::to_string(to_int(c.exit_code))
                            : std::string())
                    << "\n";
            if (!c.hint.empty())
                out << "                        hint: " << c.hint << "\n";
        }
    }
    int warnings = 0;
    for (const DoctorCheck& c : r.checks)
        warnings += c.status == CheckStatus::warn;
    char took[32];
    std::snprintf(took, sizeof(took), "%.1f s", r.duration_ms / 1000.0);
    out << "\nResult: "
        << (r.status == CheckStatus::fail ? "FAILED (exit " + std::to_string(to_int(r.exit_code)) +
                                                ", " + std::string(name(r.exit_code)) + ")"
                                          : (warnings ? "OK with " + std::to_string(warnings) +
                                                            " warning" + (warnings == 1 ? "" : "s")
                                                      : std::string("OK")))
        << " in " << took << "\n";
    return out.str();
}

json to_json(const DoctorReport& r) {
    json checks = json::array();
    for (const DoctorCheck& c : r.checks) {
        checks.push_back({{"name", c.name},
                          {"status", to_string(c.status)},
                          {"summary", c.summary},
                          {"code", c.code ? json(*c.code) : json(nullptr)},
                          {"hint", c.hint.empty() ? json(nullptr) : json(c.hint)},
                          {"exit_code", c.status == CheckStatus::fail ? json(to_int(c.exit_code))
                                                                      : json(nullptr)},
                          {"duration_ms", c.duration_ms},
                          {"details", c.details}});
    }
    auto opt = [](const std::optional<std::string>& v) { return v ? json(*v) : json(nullptr); };
    return {
        {"doctor_schema_version", kDoctorSchemaVersion},
        {"tool",
         {{"name", "edgebench"},
          {"version", version()},
          {"git_commit", opt(r.build.git_commit)},
          {"git_dirty", r.build.git_dirty ? json(*r.build.git_dirty) : json(nullptr)},
          {"build_type", r.build.build_type.empty() ? json(nullptr) : json(r.build.build_type)},
          {"compiler", r.build.compiler},
          {"target_arch", pins::kTargetArch}}},
        {"versions",
         {{"onnxruntime",
           {{"pinned", r.versions.onnxruntime_pinned},
            {"loaded", opt(r.versions.onnxruntime_loaded)}}},
          {"qnn_ep", r.versions.qnn_in_build ? json{{"pinned", r.versions.qnn_ep_pinned},
                                                    {"reported", opt(r.versions.qnn_ep_reported)}}
                                             : json(nullptr)},
          {"qairt_sdk", r.versions.qnn_in_build
                            ? json{{"pinned", r.versions.qairt_sdk_pinned},
                                   {"reported", opt(r.versions.qairt_sdk_reported)}}
                            : json(nullptr)}}},
        {"bundle_dir", path_to_utf8(r.bundle_dir)},
        {"system", to_json(r.system)},
        {"checks", checks},
        {"overall", {{"status", to_string(r.status)}, {"exit_code", to_int(r.exit_code)}}},
        {"duration_ms", r.duration_ms},
    };
}

}  // namespace edgebench
