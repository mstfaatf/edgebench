#include "edgebench/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include <windows.h>

#include "edgebench/bundle_modules.hpp"
#include "edgebench/ep_context_cache.hpp"
#include "edgebench/interrupt.hpp"
#include "edgebench/npu_options.hpp"
#include "edgebench/qnn_backend_info.hpp"
#include "edgebench/run_identity.hpp"
#include "edgebench/runtime.hpp"
#include "edgebench/scheduler.hpp"

namespace edgebench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string trial_label(int trial, int trials) {
    return "trial " + std::to_string(trial + 1) + " of " + std::to_string(trials);
}

Error with_context(Error e, const std::string& context) {
    e.message = context + ": " + e.message;
    return e;
}

std::vector<std::vector<std::byte>> copy_outputs(const BenchSession& s) {
    std::vector<std::vector<std::byte>> out;
    for (std::size_t i = 0; i < s.outputs().size(); ++i) {
        const TensorView v = s.output(i);
        out.emplace_back(v.bytes.begin(), v.bytes.end());
    }
    return out;
}

std::optional<std::size_t> argmax_float(const std::vector<std::byte>& bytes) {
    const std::size_t n = bytes.size() / sizeof(float);
    if (n == 0)
        return std::nullopt;
    std::size_t best = 0;
    float best_value = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, bytes.data() + i * sizeof(float), sizeof(v));
        if (i == 0 || v > best_value) {
            best = i;
            best_value = v;
        }
    }
    return best;
}

Error interrupted_error(const std::string& where) {
    return Error{ExitCode::interrupted, "interrupted",
                 "interrupted by " + stop_reason() + " (" + where + ")"};
}

// What a trial reports besides its result.
struct TrialOutcome {
    std::vector<std::vector<std::byte>> final_outputs;  // last iteration's outputs
    std::string failed_stage;                           // error.stage if it failed
    int discarded_iterations = -1;  // measured iterations dropped by an interrupt
};

// Calls the test hook, then reports whether a stop was requested.
bool should_stop(const StopHook& hook, StopPoint point, int trial, int iteration) {
    if (hook)
        hook(point, trial, iteration);
    return stop_requested();
}

// Every ONNX Runtime / QNN module in the process must come from the directory of
// the onnxruntime.dll that was loaded (the bundle). Checked after the provider was
// created (its EP library registered) and after each session is created, when
// execution providers have pulled in their libraries; every check is recorded
// (verification_details.module_checks). On failure the full module list goes into
// the result, so the foreign module is recorded.
Result<void> check_loaded_modules(BenchmarkResult& result, const char* stage,
                                  std::optional<int> trial_index) {
    const std::filesystem::path& dll = result.runtime.onnxruntime.dll_path;
    const std::filesystem::path bundle = dll.empty() ? executable_dir() : dll.parent_path();
    BundleModuleCheck check = check_bundle_modules(bundle);
    result.execution.module_checks.push_back({stage, trial_index, check.outside});
    if (check.ok())
        return {};
    result.runtime.loaded_modules = std::move(check.modules);
    return check.result();
}

// What ONNX Runtime logs when an EP rates a compiled (EP context) model as usable
// but not optimal for the device.
constexpr std::string_view kRecompileMessage = "reports compiled model is supported but suboptimal";

std::filesystem::path utf8_path(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

// --- EP context cache --------------------------------------------------------

std::string utc_now() {
    return utc_timestamp(std::chrono::system_clock::now());
}

// A session from an EP context model that ONNX Runtime could not create: the
// context is unusable (a backend that failed to load keeps its own error).
Error context_load_failed(Error e, const std::filesystem::path& context_model) {
    if (e.exit_code == ExitCode::ort_load_failure)
        return e;
    e.exit_code = ExitCode::ort_runtime_error;
    e.kind = "ep_context_load_failed";
    e.message = "cannot create a session from the EP context model " + path_to_utf8(context_model) +
                " (" + e.message +
                "); the context may be corrupt or truncated: rebuild it with --ep-cache build";
    e.unsupported_nodes.reset();
    return e;
}

// The compatibility info of an EP context model the run built or loaded: the
// QNN EP stores the HTP arch it compiled for there (read from the device's
// platform info), so it fills runtime.qnn.htp_arch if nothing did yet. With
// `record`, the info and the EP's verdict go into config.ep_context_cache.
void apply_context_compatibility(Provider& provider, BenchmarkResult& r,
                                 const std::filesystem::path& context_model, bool record) {
    std::optional<QnnBackendInfo>& qnn = r.runtime.provider.qnn;
    if (!record && (!qnn || qnn->htp_arch))
        return;
    const ContextCompatibility c = provider.context_compatibility(context_model);
    if (record)
        r.ep_cache.compatibility = c;
    if (!qnn || qnn->htp_arch || !c.info)
        return;
    if (const std::optional<QnnCompatibilityInfo> info = parse_qnn_compatibility_info(*c.info)) {
        if (std::optional<std::string> arch = htp_arch_name(*info)) {
            qnn->htp_arch = std::move(arch);
            qnn->htp_arch_source = std::string(kSourceEpCompatibilityInfo);
        }
    }
}

// The key fields of this run's EP context (ep_context_cache.hpp).
CacheKeyFields cache_key_fields(const BenchmarkResult& r, const RunConfig& config) {
    const ProviderDescription& p = r.runtime.provider;
    CacheKeyFields k;
    k.model_sha256 = r.model.file.sha256;
    k.ort_version = r.runtime.onnxruntime.ort_version.empty()
                        ? std::nullopt
                        : std::optional<std::string>(r.runtime.onnxruntime.ort_version);
    k.ep_version = p.ep_version;
    if (p.qnn) {
        k.qnn_sdk_version = p.qnn->sdk_version;
        if (const std::optional<ModuleFileVersion> v =
                read_module_file_version(utf8_path(p.qnn->backend_path)))
            k.qnn_backend_file_version = v->product_version;
    }
    k.ep_options = p.ep_options;
    k.session_config = compile_session_config(p.session_config);
    k.graph_optimization_level = std::string(schema_name(config.graph_opt));
    if (p.ep_device)
        k.device = CacheDevice{p.ep_device->device.device_type, p.ep_device->device.vendor_id,
                               p.ep_device->device.device_id};
    return k;
}

// Deletes an entry directory the run created (writing its manifest first, so that
// only what the build wrote is deleted). Errors are ignored: the run already failed.
void discard_entry(const std::filesystem::path& dir, const std::string& key_id,
                   const nlohmann::json& key, const std::string& context_model) {
    if (write_cache_manifest(dir, key_id, key, context_model, utc_now()))
        (void)delete_cache_entry(dir.parent_path(), dir);
}

struct BuiltContext {
    double build_ms = 0;
    CacheManifest manifest;
};

// Builds `model`'s EP context in the new directory `dir` (a direct subdirectory of
// the cache directory) and writes its manifest. On failure the directory's files
// are deleted again.
Result<BuiltContext> build_context(Provider& provider, const RunConfig& config, BenchmarkResult& r,
                                   const std::filesystem::path& dir, const std::string& key_id,
                                   const nlohmann::json& key, std::optional<int> trial) {
    std::error_code ec;
    if (!std::filesystem::create_directory(dir, ec)) {
        return Error{ExitCode::output_error, "ep_cache_dir_not_writable",
                     "cannot create " + path_to_utf8(dir) + " in the EP context cache: " +
                         (ec ? ec.message() : std::string("it already exists"))};
    }
    const std::string name = context_model_name(r.model.file.path);
    const std::filesystem::path context_model = dir / utf8_path(name);
    Result<std::unique_ptr<BenchSession>> session =
        provider.create_context_building_session(r.model.file.path, config, context_model);
    if (!session) {
        discard_entry(dir, key_id, key, name);
        return session.error();
    }
    const double ms = session.value()->session_create_ms();
    Result<void> modules = check_loaded_modules(r, "after_untimed_session", trial);
    session.value().reset();
    if (!modules) {
        discard_entry(dir, key_id, key, name);
        return modules.error();
    }
    if (!std::filesystem::is_regular_file(context_model, ec)) {
        discard_entry(dir, key_id, key, name);
        return Error{
            ExitCode::ort_runtime_error, "ep_context_not_written",
            "the session with ep.context_enable = 1 did not write " + path_to_utf8(context_model)};
    }
    Result<CacheManifest> manifest = write_cache_manifest(dir, key_id, key, name, utc_now());
    if (!manifest) {
        discard_entry(dir, key_id, key, name);
        return manifest.error();
    }
    return BuiltContext{ms, std::move(manifest).value()};
}

// One untimed session: created, checked, run once; its outputs.
Result<std::vector<std::vector<std::byte>>> outputs_of(Provider& provider, const RunConfig& config,
                                                       BenchmarkResult& r,
                                                       const std::filesystem::path& model) {
    Result<std::unique_ptr<BenchSession>> session = provider.create_session(model, config);
    if (!session)
        return session.error();
    if (Result<void> m = check_loaded_modules(r, "after_untimed_session", std::nullopt); !m)
        return m.error();
    if (Result<void> run = session.value()->run_once(); !run)
        return run.error();
    return copy_outputs(*session.value());
}

// Before the trials of an NPU run that uses EP contexts: the cache directory, and
// for --ep-cache build|use the key, stale entries, the cache entry (built if
// needed) and the check that a session from it computes exactly what a session
// compiled from the model computes. Fills result.ep_cache.
Result<void> prepare_ep_cache(Provider& provider, const RunConfig& config, BenchmarkResult& r,
                              const ProgressFn& progress, std::string& stage) {
    EpContextCacheResult& c = r.ep_cache;
    c.mode = config.ep_cache;
    stage = "configuration";
    Result<std::filesystem::path> dir = config.ep_cache_dir
                                            ? Result<std::filesystem::path>(*config.ep_cache_dir)
                                            : default_ep_cache_dir();
    if (!dir)
        return dir.error();
    Result<std::filesystem::path> ready = prepare_ep_cache_dir(dir.value());
    if (!ready)
        return ready.error();
    c.cache_dir = ready.value();
    if (!c.enabled()) {
        c.state = "disabled";
        return {};
    }
    const std::filesystem::path& cache_dir = *c.cache_dir;
    const CacheKeyFields fields = cache_key_fields(r, config);
    const std::string key_id = cache_key_id(fields);
    const nlohmann::json key = cache_key_json(fields);
    c.key_id = key_id;
    c.key = key;

    Result<std::vector<std::filesystem::path>> stale =
        delete_stale_cache_entries(cache_dir, fields);
    if (!stale)
        return stale.error();
    c.deleted_stale = std::move(stale).value();

    CacheLookup found = lookup_cache_entry(cache_dir, key_id, key);
    if (found.state == CacheEntryState::foreign) {
        return Error{ExitCode::output_error, "ep_cache_entry_foreign",
                     found.problem +
                         "; edgebench only deletes files listed in a manifest it wrote, so remove "
                         "the directory or use another --ep-cache-dir"};
    }
    bool replaced = false;
    if (config.ep_cache == EpCacheMode::build && found.state != CacheEntryState::missing) {
        Result<std::vector<std::filesystem::path>> d =
            delete_cache_entry(cache_dir, found.entry_dir);
        if (!d)
            return d.error();
        replaced = true;
        found.state = CacheEntryState::missing;
    }
    stage = "session_creation";
    if (found.state == CacheEntryState::corrupt) {
        return Error{ExitCode::ort_runtime_error, "ep_context_corrupt",
                     "the EP context cache entry " + path_to_utf8(found.entry_dir) +
                         " cannot be used: " + found.problem +
                         "; rebuild it with --ep-cache build"};
    }

    CacheManifest manifest;
    if (found.state == CacheEntryState::valid) {
        c.hit = true;
        c.state = "loaded";
        manifest = *found.manifest;
    } else {
        c.hit = false;
        if (progress)
            progress("building the EP context cache (key " + key_id.substr(0, 12) + ")");
        const std::filesystem::path staging =
            cache_dir / (".build-" + key_id.substr(0, 16) + "-" +
                         std::to_string(GetCurrentProcessId()) + "-" + r.run_id.substr(0, 8));
        Result<BuiltContext> built =
            build_context(provider, config, r, staging, key_id, key, std::nullopt);
        if (!built)
            return built.error();
        if (!MoveFileExW(staging.c_str(), found.entry_dir.c_str(), MOVEFILE_WRITE_THROUGH)) {
            const DWORD code = GetLastError();
            (void)delete_cache_entry(cache_dir, staging);
            return Error{ExitCode::output_error, "ep_cache_dir_not_writable",
                         "cannot move the new EP context cache entry to " +
                             path_to_utf8(found.entry_dir) + " (Windows error " +
                             std::to_string(code) + ")"};
        }
        r.timing.session_init.context_build_ms = built.value().build_ms;
        manifest = std::move(built.value().manifest);
        c.state = replaced || !c.deleted_stale.empty() ? "regenerated" : "generated";
    }
    c.files = manifest.files;
    c.context_path = found.entry_dir / utf8_path(manifest.context_model);
    apply_context_compatibility(provider, r, *c.context_path, true);

    // A session from the cache must compute exactly what a compiled one computes.
    if (progress)
        progress("checking that a session from the EP context cache matches a compiled one");
    Result<std::vector<std::vector<std::byte>>> compiled =
        outputs_of(provider, config, r, r.model.file.path);
    if (!compiled)
        return compiled.error();
    Result<std::vector<std::vector<std::byte>>> cached =
        outputs_of(provider, config, r, *c.context_path);
    if (!cached)
        return context_load_failed(cached.error(), *c.context_path);
    c.verified_equal_outputs = compiled.value() == cached.value();
    if (!*c.verified_equal_outputs) {
        stage = "measurement";
        return Error{ExitCode::internal_error, "ep_context_outputs_differ",
                     "a session created from the EP context model " +
                         path_to_utf8(*c.context_path) +
                         " gave outputs that are not bit-identical to a session compiled from the "
                         "model, for the same input"};
    }
    return {};
}

// The three --init-breakdown session creations of one trial, each after the same
// idle period: cold (from the model), context build (into a scratch entry
// directory of this trial) and cached (from that scratch context, which is then
// deleted).
Result<TrialSessionInit> measure_init_breakdown(
    Provider& provider, const RunConfig& config, BenchmarkResult& r, int trial,
    const std::function<Result<void>(const char*)>& idle) {
    TrialSessionInit init;
    const std::filesystem::path& cache_dir = *r.ep_cache.cache_dir;
    const CacheKeyFields fields = cache_key_fields(r, config);
    const std::string key_id = cache_key_id(fields);
    const nlohmann::json key = cache_key_json(fields);

    if (Result<void> i = idle("cold session creation"); !i)
        return i.error();
    {
        Result<std::unique_ptr<BenchSession>> s =
            provider.create_session(r.model.file.path, config);
        if (!s)
            return s.error();
        init.cold_ms = s.value()->session_create_ms();
        if (Result<void> m = check_loaded_modules(r, "after_untimed_session", trial); !m)
            return m.error();
    }

    if (Result<void> i = idle("context build"); !i)
        return i.error();
    const std::filesystem::path scratch =
        cache_dir / ("scratch-" + r.run_id.substr(0, 8) + "-t" + std::to_string(trial + 1));
    Result<BuiltContext> built = build_context(provider, config, r, scratch, key_id, key, trial);
    if (!built)
        return built.error();
    init.context_build_ms = built.value().build_ms;
    const std::filesystem::path context_model =
        scratch / utf8_path(built.value().manifest.context_model);
    apply_context_compatibility(provider, r, context_model, false);

    Result<void> step = idle("cached session creation");
    if (step) {
        Result<std::unique_ptr<BenchSession>> s = provider.create_session(context_model, config);
        if (!s) {
            step = context_load_failed(s.error(), context_model);
        } else {
            init.cached_init_ms = s.value()->session_create_ms();
            step = check_loaded_modules(r, "after_untimed_session", trial);
        }
    }
    Result<std::vector<std::filesystem::path>> removed = delete_cache_entry(cache_dir, scratch);
    if (!step)
        return step.error();
    if (!removed)
        return removed.error();
    return init;
}

// The --profile-accelerator pass: a session with QNN profiling from `source`
// (what the timed sessions were created from), --warmup inferences, then the
// requested ones; released, so the EP writes its profile, which is then read.
// Untimed: nothing here touches the trials or their statistics.
Result<ProfiledPassResult> run_profiled_pass(Provider& provider, const RunConfig& config,
                                             BenchmarkResult& r,
                                             const std::filesystem::path& source,
                                             const StopHook& hook) {
    ProfiledPassResult p;
    p.warmup_iterations = config.warmup;
    p.iterations = config.effective_profile_iterations();
    std::error_code ec;
    const std::filesystem::path temp = std::filesystem::temp_directory_path(ec);
    p.staging_dir = temp / ("edgebench-profile-" + r.run_id);
    if (ec || !std::filesystem::create_directories(p.staging_dir, ec)) {
        return Error{ExitCode::output_error, "profile_dir_not_writable",
                     "cannot create the directory for the QNN profile " +
                         path_to_utf8(p.staging_dir) +
                         (ec ? ": " + ec.message() : std::string(": it already exists"))};
    }
    auto fail = [&p](Error e) -> Result<ProfiledPassResult> {
        std::error_code ignored;
        std::filesystem::remove_all(p.staging_dir, ignored);
        return e;
    };
    const std::filesystem::path csv = p.staging_dir / "qnn_profile.csv";
    {
        Result<std::unique_ptr<BenchSession>> session =
            provider.create_profiling_session(source, config, csv);
        if (!session) {
            return fail(source == r.model.file.path ? session.error()
                                                    : context_load_failed(session.error(), source));
        }
        if (Result<void> m = check_loaded_modules(r, "profiled_session", std::nullopt); !m)
            return fail(m.error());
        const int total = p.warmup_iterations + p.iterations;
        for (int i = 0; i < total; ++i) {
            if (should_stop(hook, StopPoint::profiled_pass, 0, i))
                return fail(interrupted_error("profiled accelerator pass, inference " +
                                              std::to_string(i + 1)));
            if (Result<void> run = session.value()->run_once(); !run)
                return fail(with_context(
                    run.error(), "profiled accelerator pass, inference " + std::to_string(i + 1)));
        }
    }  // released: the EP writes the profile now
    std::string text;
    {
        std::ifstream in(csv, std::ios::binary);
        if (!in) {
            return fail(Error{ExitCode::internal_error, "qnn_profile_missing",
                              "the QNN EP did not write its profile " + path_to_utf8(csv)});
        }
        std::ostringstream content;
        content << in.rdbuf();
        text = content.str();
    }
    Result<std::vector<QnnProfileEvent>> events = parse_qnn_profile_csv(text);
    if (!events)
        return fail(events.error());
    p.profile = summarize_qnn_profile(events.value(), p.warmup_iterations);
    for (const std::filesystem::directory_entry& e :
         std::filesystem::directory_iterator(p.staging_dir, ec)) {
        if (e.is_regular_file(ec))
            p.files.push_back(e.path());
    }
    std::sort(p.files.begin(), p.files.end());
    return p;
}

// The HTP backend library of the QNN EP, whose version resource is recorded.
constexpr std::string_view kQnnHtpBackendFile = "QnnHtp.dll";

// One trial: fresh session, warm-up, measured iterations. Stop requests are
// honoured before and after session creation and between iterations (never
// inside the timed bracket).
Result<TrialResult> run_trial(Provider& provider, const RunConfig& config, const ScheduledTrial& st,
                              const std::filesystem::path& source, BenchmarkResult& result,
                              const StopHook& hook, TrialOutcome& out) {
    const std::string where = trial_label(st.trial_index, config.trials);
    out.failed_stage = "session_creation";
    if (should_stop(hook, StopPoint::before_session, st.trial_index, 0))
        return interrupted_error(where + ", before session creation");
    Result<std::unique_ptr<BenchSession>> created = provider.create_session(source, config);
    if (!created) {
        return with_context(source == result.model.file.path
                                ? created.error()
                                : context_load_failed(created.error(), source),
                            where);
    }
    std::unique_ptr<BenchSession> session = std::move(created).value();
    result.known.session = true;
    if (Result<void> modules =
            check_loaded_modules(result, "after_session_creation", st.trial_index);
        !modules)
        return with_context(modules.error(), where);
    if (should_stop(hook, StopPoint::after_session, st.trial_index, 0)) {
        out.discarded_iterations = 0;
        return interrupted_error(where + ", after session creation");
    }

    TrialResult trial;
    trial.trial_index = st.trial_index;
    trial.sequence = st.sequence;
    trial.session_create_ms = session->session_create_ms();
    trial.session_create_working_set_delta_bytes =
        session->session_create_working_set_delta_bytes();

    if (st.sequence == 0) {
        // Snapshot while a session is alive: EP libraries can be unloaded with it.
        result.runtime.loaded_modules = list_ort_modules();
        if (result.runtime.provider.qnn) {
            if (const LoadedModule* htp =
                    find_loaded_module(result.runtime.loaded_modules, kQnnHtpBackendFile))
                result.runtime.qnn_backend_file_version = read_module_file_version(htp->path);
        }
        result.batch_size = session->batch_size();
        result.model.inputs = session->inputs();
        result.model.outputs = session->outputs();
    }

    auto run = [&session, &where](const char* stage, int iteration) -> Result<void> {
        if (Result<void> r = session->run_once(); !r) {
            return with_context(
                r.error(), where + ", " + stage + " iteration " + std::to_string(iteration + 1));
        }
        return {};
    };

    out.failed_stage = "warmup";
    for (int i = 0; i < config.warmup; ++i) {
        if (should_stop(hook, StopPoint::warmup, st.trial_index, i)) {
            out.discarded_iterations = 0;
            return interrupted_error(where + ", warm-up iteration " + std::to_string(i + 1));
        }
        const Clock::time_point t0 = Clock::now();
        if (Result<void> r = run("warm-up", i); !r)
            return r.error();
        if (i == 0)
            trial.first_inference_ms = ms_between(t0, Clock::now());
    }

    out.failed_stage = "measurement";
    trial.latencies_ms.reserve(static_cast<std::size_t>(config.iterations));
    // Background load across the measured iterations (other processes only), read
    // outside the timed brackets.
    const LoadProbe load_probe = start_load_probe();
    for (int i = 0; i < config.iterations; ++i) {
        if (should_stop(hook, StopPoint::measured, st.trial_index, i)) {
            out.discarded_iterations = i;
            return interrupted_error(where + ", measured iteration " + std::to_string(i + 1));
        }
        const Clock::time_point t0 = Clock::now();
        const Result<void> r = session->run_once();
        const Clock::time_point t1 = Clock::now();
        if (!r) {
            return with_context(r.error(), where + ", measured iteration " + std::to_string(i + 1));
        }
        trial.latencies_ms.push_back(ms_between(t0, t1));
    }
    trial.during = finish_load_probe(load_probe);

    for (std::size_t i = 0; i < session->outputs().size(); ++i) {
        if (const auto bad = first_non_finite(session->output(i))) {
            return Error{ExitCode::internal_error, "non_finite_output",
                         where + ": output '" + session->outputs()[i].name +
                             "' has a non-finite value (NaN or infinity) at element " +
                             std::to_string(*bad)};
        }
    }
    out.final_outputs = copy_outputs(*session);
    trial.stats = compute_stats(trial.latencies_ms, session->batch_size());
    return trial;  // the session is released here
}

// Compares a trial's final outputs with trial 0's.
Result<void> check_determinism(const BenchmarkResult& result, int trial,
                               const std::vector<std::vector<std::byte>>& reference,
                               const std::vector<std::vector<std::byte>>& outputs,
                               std::vector<ResultWarning>& warnings) {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == reference[i])
            continue;
        const TensorInfo& info = result.model.outputs[i];
        const std::string what = "output '" + info.name + "' of trial " + std::to_string(trial + 1);
        if (info.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            argmax_float(outputs[i]) == argmax_float(reference[i])) {
            warnings.push_back(
                {"nondeterministic_outputs",
                 what + " is not bit-identical to trial 1 for the same input (same top-1)",
                 std::nullopt});
            continue;
        }
        return Error{
            ExitCode::internal_error, "inconsistent_outputs",
            what + " differs from trial 1 for the same input" +
                (info.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? " (different top-1)" : "")};
    }
    return {};
}

// "CPUExecutionProvider 1, QNNExecutionProvider 2"
std::string describe_placement(const NodeAssignment& p) {
    std::string out;
    for (const auto& [ep, n] : p.nodes_by_ep)
        out += (out.empty() ? "" : ", ") + ep + " " + std::to_string(n);
    return out;
}

// Nodes ONNX Runtime places on the QNN EP: each one is a fused partition.
int qnn_partitions(const NodeAssignment& p) {
    const auto it = p.nodes_by_ep.find("QNNExecutionProvider");
    return it == p.nodes_by_ep.end() ? 0 : it->second;
}

void add_unavailable(BenchmarkResult& r, const char* path, const char* message) {
    r.warnings.push_back({"value_unavailable", message, std::string(path)});
}

std::string fmt(double v, int decimals = 4) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string fmt_opt(const std::optional<double>& v) {
    return v ? fmt(*v) : "-";
}

}  // namespace

const std::vector<double>& ProfiledPassResult::accelerator_ms() const {
    static const std::vector<double> kNone;
    const auto it = profile.execute_ms.find(std::string(kAcceleratorExecuteEvent));
    return it == profile.execute_ms.end() ? kNone : it->second;
}

std::chrono::milliseconds idle_before_trial(const RunConfig& config, int sequence) {
    if (config.cooldown_ms > 0)
        return std::chrono::milliseconds(config.cooldown_ms);
    return sequence == 0 ? std::chrono::milliseconds(config.load_sample_ms)
                         : std::chrono::milliseconds(0);
}

std::string generated_input_description(const RunConfig& config) {
    const std::string data = config.input == InputKind::random
                                 ? "random values in [0, 1) from SplitMix64 (seed " +
                                       std::to_string(config.effective_seed()) + ")"
                                 : "the constant " + fmt(config.effective_constant_value(), 6);
    return "No preprocessing: inputs are generated (" + data +
           ") once per session, before timing, and reused for every iteration.";
}

void set_failure(BenchmarkResult& result, Error error, std::string stage) {
    result.failure = RunFailure{std::move(error), std::move(stage)};
}

BenchmarkResult begin_result(const RunConfig& config) {
    BenchmarkResult r;
    r.schema_version = std::string(results_schema_version());
    r.run_id = new_run_id();
    r.started_at = std::chrono::system_clock::now();
    r.timestamp_utc = utc_timestamp(r.started_at);
    r.label = config.label;
    r.tool_version = std::string(version());
    r.tool = build_info();
    r.config = config;
    r.input_preprocessing = generated_input_description(config);
    r.timing.timer = steady_clock_info();
    r.system = query_system_info();
    r.system.power = query_power_state();

    // Model file facts first: reading the file also warms the file cache equally
    // for every trial.
    Result<ModelFileInfo> file = read_model_file_info(config.model);
    if (!file) {
        set_failure(r, file.error(), "model_load");
        return r;
    }
    r.model.file = std::move(file).value();
    r.model.name = path_to_utf8(r.model.file.path.stem());
    Result<OnnxGraphSummary> graph = inspect_onnx_file(r.model.file.path);
    if (graph) {
        const OnnxGraphSummary& g = graph.value();
        r.model.opset = g.opset;
        r.model.ir_version = g.ir_version;
        r.model.precision = infer_precision(g);
        r.model.normalization = normalization_from_metadata(g);
        r.model.quantization_record = quantization_from_metadata(g);
        if (!g.producer_name.empty())
            r.model.export_tool =
                g.producer_name + (g.producer_version.empty() ? "" : " " + g.producer_version);
        // Tensors as declared in the file; replaced by the session's once one exists.
        r.model.inputs = g.inputs;
        r.model.outputs = g.outputs;
        if (!g.inputs.empty() && !g.inputs.front().shape.empty() &&
            g.inputs.front().shape.front() > 0) {
            r.batch_size = g.inputs.front().shape.front();
        }
        r.known.model = !g.inputs.empty() && !g.outputs.empty();
        r.graph = std::move(graph).value();
    }
    return r;
}

BenchmarkResult failed_load_result(const RunConfig& config, const OrtLoadResult& load) {
    BenchmarkResult r = begin_result(config);
    if (!r.failure) {
        set_failure(r,
                    Error{ExitCode::ort_load_failure, std::string(to_string(load.error_kind)),
                          load.message},
                    "environment");
    }
    finish_result(r);
    return r;
}

BenchmarkResult run_benchmark(Runtime& runtime, const RunConfig& config, const ProgressFn& progress,
                              const StopHook& hook) {
    // A model the NPU cannot run in the requested precision is a usage error;
    // report it before registering the QNN EP and selecting the device.
    if (config.provider == ProviderKind::npu) {
        if (Result<OnnxGraphSummary> graph = inspect_onnx_file(config.model); graph) {
            if (Result<std::optional<std::string>> precision =
                    requested_compute_precision(config, infer_precision(graph.value()));
                !precision) {
                BenchmarkResult r = begin_result(config);
                if (!r.failure)
                    set_failure(r, precision.error(), "model_load");
                finish_result(r);
                return r;
            }
        }
    }
    Result<std::unique_ptr<Provider>> provider = make_provider(runtime, config);
    if (!provider) {
        BenchmarkResult r = begin_result(config);
        r.runtime.onnxruntime = runtime.load_result();
        r.known.runtime = true;
        try {
            r.runtime.ep_devices = runtime.ep_devices();
        } catch (const Ort::Exception&) {
            // stays null: the devices could not be enumerated
        }
        if (!r.failure)
            set_failure(r, provider.error(), "ep_registration");
        finish_result(r);
        return r;
    }
    return run_benchmark(runtime, *provider.value(), config, progress, hook);
}

BenchmarkResult run_benchmark(Runtime& runtime, Provider& provider, const RunConfig& config,
                              const ProgressFn& progress, const StopHook& hook) {
    auto report = [&progress](const std::string& message) {
        if (progress)
            progress(message);
    };

    BenchmarkResult r = begin_result(config);
    r.runtime.onnxruntime = runtime.load_result();
    r.known.runtime = true;
    r.runtime.provider = provider.describe();
    r.known.provider = true;
    if (const std::optional<int> logical = r.system.cpu.logical_processors;
        logical && config.threads > *logical) {
        r.warnings.push_back(
            {"threads_exceed_logical_processors",
             std::to_string(config.threads) + " intra-op threads requested on a CPU with " +
                 std::to_string(*logical) + " logical processors; threads will compete for cores",
             std::string("/config/intra_op_threads")});
    }
    if (r.runtime.provider.qnn)
        r.timing.session_init.ep_context_state = "cold";  // compiled from the graph, no cache
    if (!r.failure) {
        if (Result<void> modules = check_loaded_modules(r, "after_provider_creation", std::nullopt);
            !modules)
            set_failure(r, modules.error(), "ep_registration");
    }
    if (!r.failure) {
        if (!r.graph && config.provider == ProviderKind::npu) {
            set_failure(r,
                        Error{ExitCode::usage_error, "invalid_onnx",
                              "cannot decode " + path_to_utf8(config.model.filename()) +
                                  ", so its precision, which decides the NPU compute precision, "
                                  "is unknown"},
                        "model_load");
        } else if (r.graph) {
            Result<std::optional<std::string>> precision =
                requested_compute_precision(config, r.model.precision);
            if (!precision) {
                set_failure(r, precision.error(), "model_load");
            } else {
                r.compute_precision = precision.value();
                r.model.precision_notes = precision_notes(r.model.precision, r.compute_precision);
                r.effective_precision =
                    effective_precision(config.provider, r.model.precision, r.compute_precision);
            }
        }
    }
    if (config.provider == ProviderKind::npu && !config.strict) {
        r.warnings.push_back({"strict_mode_off",
                              "CPU fallback is allowed (--no-strict): nodes the NPU cannot run "
                              "execute on the CPU, inside the measured time.",
                              std::string("/config/strict_mode")});
    }
    if (r.failure) {
        finish_result(r);
        return r;
    }
    try {
        r.runtime.ep_devices = runtime.ep_devices();
    } catch (const Ort::Exception& e) {
        set_failure(r,
                    ort_error("ep_enumeration_failed", "enumerating EP devices",
                              e.GetOrtErrorCode(), e.what()),
                    "environment");
        finish_result(r);
        return r;
    }

    // EP contexts: the cache directory, and for --ep-cache build|use the cache entry
    // and the cached == compiled check, before the first trial.
    const std::uint64_t log_start = runtime.log_position();
    if (config.provider == ProviderKind::npu && config.uses_ep_context()) {
        std::string stage;
        if (Result<void> prepared = prepare_ep_cache(provider, config, r, progress, stage);
            !prepared) {
            set_failure(r, prepared.error(), stage);
            finish_result(r);
            return r;
        }
    }
    if (r.ep_cache.enabled())
        r.timing.session_init.ep_context_state = "cached";  // timed sessions load the context
    const std::filesystem::path timed_source =
        r.ep_cache.context_path ? *r.ep_cache.context_path : r.model.file.path;

    std::vector<std::vector<std::byte>> reference_outputs;
    for (const ScheduledTrial& st : schedule_trials(1, config.trials)) {
        // Every trial, the first included, starts after the same idle period, so
        // session creation always meets the CPU in the same power state. With
        // --init-breakdown each of its three session creations gets that idle
        // period too, and the trial's own idle comes last, right before its timed
        // session.
        std::optional<LoadSample> idle;
        const std::string label = trial_label(st.trial_index, config.trials);
        bool first_idle = true;
        auto idle_period = [&](const std::string& before) -> Result<void> {
            const std::chrono::milliseconds wait = idle_before_trial(config, st.sequence);
            if (wait.count() <= 0)
                return {};
            report("idle " + std::to_string(wait.count()) + " ms before " + before +
                   " (sampling background CPU load)");
            // Waits on the stop event, so an interrupt ends the idle period at once.
            // Other processes only: this process's own CPU time (e.g. a session
            // still being torn down) is not background load.
            const LoadProbe probe = start_load_probe();
            const bool hook_stop = should_stop(hook, StopPoint::idle, st.trial_index, 0);
            const bool cut = hook_stop || wait_or_stop(wait);
            idle = finish_load_probe(probe).value_or(
                LoadSample{0.0, std::chrono::duration<double>(Clock::now() - probe.started).count(),
                           std::nullopt});
            if (st.sequence == 0 && first_idle && !cut)
                r.system.load = idle;
            first_idle = false;
            if (cut)
                return interrupted_error("idle period before " + before);
            return {};
        };
        std::optional<TrialSessionInit> breakdown;
        if (config.init_breakdown && config.provider == ProviderKind::npu) {
            Result<TrialSessionInit> measured = measure_init_breakdown(
                provider, config, r, st.trial_index,
                [&](const char* what) { return idle_period(std::string(what) + " of " + label); });
            if (!measured) {
                set_failure(
                    r, measured.error(),
                    measured.error().kind == "interrupted" ? "measurement" : "session_creation");
                break;
            }
            breakdown = measured.value();
        }
        if (Result<void> i = idle_period(label); !i) {
            set_failure(r, i.error(), "measurement");
            break;
        }
        report(trial_label(st.trial_index, config.trials) + ": creating session, " +
               std::to_string(config.warmup) + " warm-up + " + std::to_string(config.iterations) +
               " measured iterations");
        TrialOutcome outcome;
        Result<TrialResult> trial = run_trial(provider, config, st, timed_source, r, hook, outcome);
        if (!trial) {
            set_failure(r, trial.error(), outcome.failed_stage);
            if (outcome.discarded_iterations >= 0) {
                r.warnings.push_back(
                    {"interrupted_trial_discarded",
                     label + " was interrupted after " +
                         std::to_string(outcome.discarded_iterations) + " of " +
                         std::to_string(config.iterations) +
                         " measured iterations; the trial is not part of the result",
                     std::nullopt});
            }
            break;
        }
        std::vector<std::vector<std::byte>>& outputs = outcome.final_outputs;
        if (st.sequence == 0) {
            // ONNX Runtime accepted the model; the file decoder must have too.
            if (!r.graph) {
                set_failure(r,
                            Error{ExitCode::internal_error, "invalid_onnx",
                                  "cannot decode the model file's metadata although ONNX "
                                  "Runtime loaded it"},
                            "model_load");
                break;
            }
            reference_outputs = std::move(outputs);
        } else if (Result<void> d =
                       check_determinism(r, st.trial_index, reference_outputs, outputs, r.warnings);
                   !d) {
            set_failure(r, d.error(), "measurement");
            break;
        }
        TrialResult& done = trial.value();
        done.idle_before = idle;
        SessionInitResult& init = r.timing.session_init;
        init.timed_ms.push_back(done.session_create_ms);
        if (breakdown) {
            done.session_init = *breakdown;
            init.cold_ms.push_back(*breakdown->cold_ms);
            init.cached_ms.push_back(*breakdown->cached_init_ms);
        }
        if (r.ep_cache.enabled()) {
            init.cached_ms.push_back(done.session_create_ms);
            if (!breakdown)
                done.session_init.cached_init_ms = done.session_create_ms;
        } else {
            init.cold_ms.push_back(done.session_create_ms);
            if (!breakdown)
                done.session_init.cold_ms = done.session_create_ms;
        }
        r.timing.trials.push_back(std::move(done));
    }

    if (!r.failure) {
        // Placement last, in its own verbose session, so it cannot warm a timed session.
        r.memory_before_placement = query_process_memory();
        if (should_stop(hook, StopPoint::before_placement, 0, 0))
            set_failure(r, interrupted_error("before node placement"), "session_creation");
    }
    if (!r.failure) {
        report("recording node placement");
        Result<NodeAssignment> placement = provider.record_placement(r.model.file.path, config);
        if (!placement) {
            set_failure(r, placement.error(), "session_creation");
        } else {
            const std::string requested_ep(execution_provider_name(config.provider));
            r.execution.node_assignment = std::move(placement).value();
            r.known.placement = true;
            if (r.execution.node_assignment.modules_checked_while_alive)
                r.execution.module_checks.push_back({"placement_session", std::nullopt, {}});
            const NodeAssignment& p = r.execution.node_assignment;
            const bool all_on = p.all_on(requested_ep);
            const std::optional<QnnValidation>& qv = p.qnn_validation;
            const bool qnn_rejected = qv && (!qv->rejected_nodes.empty() || qv->failed_nodes() > 0);
            r.execution.full_device_execution_verified = all_on && !qnn_rejected;
            r.execution.verification_details =
                qv ? "Node placement from ONNX Runtime's verbose session log, and node validation "
                     "from the QNN execution provider's log of the same session, recorded in a "
                     "separate session with the same options after the timed trials. QNN "
                     "partitions (fused nodes on QNNExecutionProvider): " +
                         std::to_string(qnn_partitions(p)) + "."
                   : "Node placement from ONNX Runtime's verbose session log, recorded in a "
                     "separate session with the same options after the timed trials.";
            const std::string not_on = std::to_string(p.nodes_not_on(requested_ep)) + " of " +
                                       std::to_string(p.total_nodes) + " nodes are not on " +
                                       requested_ep;
            if (!all_on && strict_mode(config)) {
                set_failure(r,
                            Error{ExitCode::internal_error, "strict_placement_mismatch",
                                  "strict mode is on, but the placement session reports that " +
                                      not_on + " (" + describe_placement(p) +
                                      "); ONNX Runtime should have refused the session"},
                            "session_creation");
            } else if (!all_on && config.provider != ProviderKind::cpu) {
                std::string message = not_on + " and run on the CPU (" + describe_placement(p) +
                                      "), inside the measured time";
                if (qv && !qv->rejected_nodes.empty())
                    message += "; rejected by the QNN EP: " + describe_nodes(qv->rejected_nodes);
                r.warnings.push_back(
                    {"cpu_fallback_nodes", message, std::string("/execution/node_assignment")});
            } else if (!all_on) {
                r.warnings.push_back({"nodes_not_on_requested_ep", not_on,
                                      std::string("/execution/node_assignment")});
            } else if (qnn_rejected) {
                r.warnings.push_back({"qnn_validation_rejected_nodes",
                                      "every node is placed on " + requested_ep +
                                          ", but the QNN EP's validation log reports " +
                                          std::to_string(qv->failed_nodes()) +
                                          " failed node validations" +
                                          (qv->rejected_nodes.empty()
                                               ? std::string()
                                               : " (" + describe_nodes(qv->rejected_nodes) + ")"),
                                      std::string("/execution/qnn_validation")});
            }
            // The placement session itself is not interruptible; stop right after it.
            if (should_stop(hook, StopPoint::after_placement, 0, 0))
                set_failure(r, interrupted_error("after node placement"), "session_creation");
        }
    }
    // The profiled accelerator pass: last, after placement, so no timed or
    // placement session follows a profiled one.
    if (!r.failure && config.profile_accelerator) {
        report("profiled accelerator pass: " + std::to_string(config.warmup) + " warm-up + " +
               std::to_string(config.effective_profile_iterations()) +
               " inferences with QNN profiling (not timed)");
        Result<ProfiledPassResult> pass =
            run_profiled_pass(provider, config, r, timed_source, hook);
        if (!pass)
            set_failure(r, pass.error(), "profiled_pass");
        else
            r.timing.profiled_pass = std::move(pass).value();
    }
    // ONNX Runtime's verdict on the EP context models, if it logged one: a context
    // compiled for this device but not in the form the EP would compile now. The
    // outputs were checked against a compiled session, so this is a note.
    if (config.uses_ep_context()) {
        for (const std::string& message : runtime.warnings_since(log_start)) {
            if (message.find(kRecompileMessage) != std::string::npos) {
                std::string text =
                    "ONNX Runtime reported for the EP context models of this run: " + message;
                const std::optional<ContextCompatibility>& cc = r.ep_cache.compatibility;
                if (cc && cc->info) {
                    if (const std::optional<QnnCompatibilityInfo> info =
                            parse_qnn_compatibility_info(*cc->info)) {
                        if (const std::optional<std::string> why = recompilation_reason(*info))
                            text += " " + *why;
                    }
                }
                if (r.ep_cache.verified_equal_outputs == true)
                    text +=
                        " Sessions from this context gave bit-identical outputs to a session "
                        "compiled from the model.";
                r.notes.push_back({"ep_context_recompilation_suggested", text,
                                   std::string("/config/ep_context_cache")});
                break;
            }
        }
    }
    finish_result(r);
    return r;
}

void finish_result(BenchmarkResult& r) {
    const RunConfig& config = r.config;

    // Process memory: after the last trial's session was released (before the
    // placement session), or now if the run ended early.
    const ProcessMemory memory =
        r.memory_before_placement ? *r.memory_before_placement : query_process_memory();
    r.memory.peak_working_set_bytes = memory.peak_working_set_bytes;
    r.memory.peak_private_bytes = memory.peak_private_bytes;
    r.memory.working_set_after_last_trial_bytes = memory.working_set_bytes;
    {
        std::vector<double> deltas;
        for (const TrialResult& trial : r.timing.trials) {
            if (trial.session_create_working_set_delta_bytes)
                deltas.push_back(
                    static_cast<double>(*trial.session_create_working_set_delta_bytes));
        }
        if (!deltas.empty()) {
            std::sort(deltas.begin(), deltas.end());
            r.memory.session_create_working_set_delta_bytes =
                static_cast<std::int64_t>(nearest_rank(deltas, 50));
        }
    }

    // Power state at the end: anything that changed makes the run suspect.
    const PowerState power_at_start = r.system.power.value_or(PowerState{});
    r.power_at_end = query_power_state();
    if (const std::vector<std::string> changed = power_changes(power_at_start, r.power_at_end);
        !changed.empty()) {
        std::string fields;
        for (const std::string& f : changed)
            fields += (fields.empty() ? "" : ", ") + f;
        r.warnings.push_back({"power_state_changed",
                              "The power state changed during the run (" + fields +
                                  "); the result records the state at the start.",
                              std::string("/system/power")});
    }

    // Background load: every sample above the threshold, with where its CPU time went.
    for (const TrialResult& trial : r.timing.trials) {
        if (trial.idle_before && background_load_too_high(*trial.idle_before))
            r.background_load_samples.push_back(
                {"idle_before_trial", trial.trial_index, *trial.idle_before});
        if (trial.during && background_load_too_high(*trial.during))
            r.background_load_samples.push_back(
                {"measured_iterations", trial.trial_index, *trial.during});
    }
    const auto top_text = [](const LoadSample& s) {
        if (!s.attribution)
            return std::string();
        std::string out;
        for (const ProcessShare& p : s.attribution->top_processes) {
            if (out.size() > 0)
                out += ", ";
            out += p.name + " " + fmt(p.cpu_share * 100.0, 1) + "%";
            if (std::count(out.begin(), out.end(), ',') >= 2)
                break;
        }
        return out.empty() ? std::string() : " (top: " + out + ")";
    };
    // Background load: the highest idle sample.
    {
        const LoadSample* highest_sample = nullptr;
        double highest = -1.0;
        std::string when;
        for (const TrialResult& trial : r.timing.trials) {
            if (trial.idle_before && background_load_too_high(*trial.idle_before) &&
                trial.idle_before->cpu_utilization_ratio > highest) {
                highest = trial.idle_before->cpu_utilization_ratio;
                highest_sample = &*trial.idle_before;
                when = "before " + trial_label(trial.trial_index, config.trials);
            }
        }
        if (highest > kHighBackgroundLoad) {
            r.warnings.push_back({"background_cpu_load_high",
                                  "Background CPU load from other processes reached " +
                                      fmt(highest * 100.0, 1) + "% (" + when + ")" +
                                      top_text(*highest_sample) +
                                      "; other processes may have distorted the measurement.",
                                  std::string("/system/load")});
        }
    }
    // Background load during the measured iterations (this process's own CPU time
    // excluded): reported separately, because it affects the latencies directly.
    {
        const TrialResult* worst = nullptr;
        for (const TrialResult& trial : r.timing.trials) {
            if (trial.during && background_load_too_high(*trial.during) &&
                (!worst ||
                 trial.during->cpu_utilization_ratio > worst->during->cpu_utilization_ratio))
                worst = &trial;
        }
        if (worst) {
            r.warnings.push_back({"background_cpu_load_high",
                                  "Background CPU load from other processes reached " +
                                      fmt(worst->during->cpu_utilization_ratio * 100.0, 1) +
                                      "% during " + trial_label(worst->trial_index, config.trials) +
                                      " (this process's own CPU time excluded)" +
                                      top_text(*worst->during) +
                                      "; other processes may have distorted the measurement.",
                                  std::nullopt});
        }
    }

    // Statistics of the completed trials.
    TimingResult& t = r.timing;
    if (!t.trials.empty()) {
        t.first_inference_ms = t.trials.front().first_inference_ms;
        std::vector<double> sorted = t.session_init.timed_ms;
        std::sort(sorted.begin(), sorted.end());
        if (!sorted.empty())
            t.session_init.session_create_ms = nearest_rank(sorted, 50);
        std::vector<double> pooled, means;
        for (const TrialResult& trial : t.trials) {
            pooled.insert(pooled.end(), trial.latencies_ms.begin(), trial.latencies_ms.end());
            means.push_back(trial.stats.mean_ms);
        }
        t.aggregate =
            AggregateResult{compute_stats(pooled, r.batch_size), compute_trial_variation(means)};
        if (const TrialVariation& v = t.aggregate->trial_variation; trial_spread_high(v)) {
            r.warnings.push_back(
                {"trial_spread_high",
                 "Trial means range from " + fmt(v.trial_mean_min_ms) + " to " +
                     fmt(v.trial_mean_max_ms) + " ms (" +
                     (v.trial_mean_cv_ratio ? "CV " + fmt(*v.trial_mean_cv_ratio * 100.0, 1) + "%, "
                                            : std::string()) +
                     "slowest/fastest " + fmt(v.trial_mean_max_ms / v.trial_mean_min_ms, 2) +
                     "x); the trials disagree, so one aggregate number does not describe this "
                     "run.",
                 std::string("/timing/aggregate/trial_variation")});
        }
    }

    // The profiled pass: its accelerator time is part of an inference, so above the
    // timed median latency something is wrong with one of the two.
    if (t.profiled_pass) {
        const ProfiledPassResult& p = *t.profiled_pass;
        std::vector<double> acc = p.accelerator_ms();
        if (static_cast<int>(acc.size()) != p.iterations) {
            r.warnings.push_back({"profiled_pass_incomplete",
                                  "The QNN profile of the profiled accelerator pass reports " +
                                      std::to_string(acc.size()) +
                                      " accelerator execute times for " +
                                      std::to_string(p.iterations) + " profiled inferences.",
                                  std::string("/timing/profiled_pass/accelerator_ms")});
        }
        if (!acc.empty() && t.aggregate) {
            std::sort(acc.begin(), acc.end());
            const double median = nearest_rank(acc, 50);
            if (median > t.aggregate->stats.p50_ms) {
                r.warnings.push_back(
                    {"accelerator_time_exceeds_latency",
                     "The median accelerator time of the profiled pass (" + fmt(median) +
                         " ms) is above the median timed latency (" +
                         fmt(t.aggregate->stats.p50_ms) +
                         " ms), although it is one part of an inference; one of the two does not "
                         "measure what it should.",
                     std::string("/timing/profiled_pass/accelerator_ms")});
            }
        }
    }

    // Batch equivalence, from the committed reference data (needs the batch size and
    // the compute precision; recorded for runs that created a session).
    if (r.known.session && r.batch_size > 0 && r.effective_precision) {
        r.batch_equivalence = lookup_batch_equivalence(
            r.model.file.path, r.model.file.sha256,
            config.provider == ProviderKind::npu ? "npu" : "cpu", r.effective_precision->compute,
            compile_relevant_ep_options(r.runtime.provider.ep_options), r.batch_size);
        const BatchEquivalence& be = *r.batch_equivalence;
        const std::string on =
            std::string(config.provider == ProviderKind::npu ? "the NPU" : "the CPU") + " (" +
            r.effective_precision->compute + " compute)";
        const std::string batch = std::to_string(r.batch_size);
        if (be.predictions_equal_to_b1 == false) {
            r.warnings.push_back(
                {"batch_not_equivalent",
                 "At batch size " + batch +
                     " this model's predictions differ from its batch-1 "
                     "file's on " +
                     on + ": " + std::to_string(be.predictions_differing.value_or(0)) + " of " +
                     std::to_string(be.predictions_compared.value_or(0)) + " " +
                     (be.predictions_basis == "test_split_flags" ? "test windows are flagged "
                                                                   "differently"
                                                                 : "eval-quick images get a "
                                                                   "different top-1 class") +
                     ". This result measures a different computation and must not be averaged "
                     "or compared with batch-1 results as the same one.",
                 std::string("/execution/batch_equivalence/predictions_equal_to_b1")});
        } else if (be.rounding_only()) {
            r.notes.push_back({"batch_rounding_only",
                               "At batch size " + batch +
                                   " this model's outputs differ from its batch-1 "
                                   "file's on " +
                                   on + " in their last bits only: no prediction changes (" +
                                   std::to_string(be.predictions_compared.value_or(0)) + " " +
                                   (be.predictions_basis == "test_split_flags" ? "test windows"
                                                                               : "eval-quick "
                                                                                 "images") +
                                   " compared).",
                               std::string("/execution/batch_equivalence/bitwise_equal_to_b1")});
        } else if (!be.bitwise_equal_to_b1) {
            if (be.options_not_covered) {
                r.notes.push_back(
                    {"batch_equivalence_options_not_covered",
                     "Batch equivalence was measured for this model file with other "
                     "compile-relevant EP options than this run's (" +
                         describe_compile_options(
                             compile_relevant_ep_options(r.runtime.provider.ep_options)) +
                         "). Options can change how the backend compiles the graph, so those "
                         "facts are not reused here.",
                     std::string("/execution/batch_equivalence/bitwise_basis")});
            }
            r.warnings.push_back({"value_unavailable",
                                  "Whether batch size " + batch +
                                      " computes the same outputs as batch 1 is "
                                      "unknown: " +
                                      be.reason + ".",
                                  std::string("/execution/batch_equivalence/bitwise_equal_to_b1")});
        } else if (!be.predictions_equal_to_b1) {
            r.warnings.push_back(
                {"value_unavailable",
                 "Batch size " + batch +
                     " does not compute bit-identical outputs to batch 1, "
                     "and whether its predictions differ is unknown: " +
                     be.reason + ".",
                 std::string("/execution/batch_equivalence/predictions_equal_to_b1")});
        }
    }

    // Values that could not be determined.
    if (!power_at_start.source)
        add_unavailable(r, "/system/power/source", "The power source could not be determined.");
    if (!power_at_start.power_mode) {
        add_unavailable(r, "/system/power/power_mode", "The Windows power mode could not be read.");
    } else if (!power_at_start.power_mode_known) {
        r.warnings.push_back({"unknown_power_mode",
                              "The power mode overlay " + *power_at_start.power_mode +
                                  " is not a known Windows power mode; it is recorded as its GUID.",
                              std::string("/system/power/power_mode")});
    }
    if (!power_at_start.power_plan)
        add_unavailable(r, "/system/power/power_plan", "The active power plan could not be read.");
    if (!power_at_start.battery_saver)
        add_unavailable(r, "/system/power/battery_saver",
                        "The battery saver state could not be read.");
    if (!r.system.load) {
        add_unavailable(r, "/system/load",
                        idle_before_trial(config, 0).count() == 0
                            ? "Background CPU load was not sampled (--load-sample-ms 0)."
                            : "Background CPU load was not sampled: the run ended before the "
                              "first trial.");
    }
    if (!r.memory.session_create_working_set_delta_bytes) {
        add_unavailable(r, "/memory/session_create_working_set_delta_bytes",
                        t.trials.empty() ? "No trial completed."
                                         : "The process working set could not be read.");
    }
    if (r.known.provider && r.runtime.provider.ep_device &&
        !r.runtime.provider.ep_device->driver_version) {
        add_unavailable(r, "/runtime/ep_device/driver_version",
                        "The device driver version could not be determined.");
    }
    if (r.known.provider && r.runtime.provider.qnn) {
        const QnnBackendInfo& q = *r.runtime.provider.qnn;
        if (!q.sdk_version)
            add_unavailable(r, "/runtime/qnn/sdk_version",
                            "The QNN EP did not report its backend's build id, from which the "
                            "SDK version is read.");
        if (!q.backend_build)
            add_unavailable(r, "/runtime/qnn/backend_build",
                            "The QNN EP did not report its backend's build id.");
        if (!q.htp_arch)
            add_unavailable(r, "/runtime/qnn/htp_arch",
                            config.uses_ep_context()
                                ? "The HTP architecture was not found in the compatibility info "
                                  "of this run's EP context."
                                : "The QNN EP reports the HTP architecture only in the "
                                  "compatibility info of an EP context it compiles; this run "
                                  "used none (--ep-cache off, no --init-breakdown).");
        if (!q.soc_model)
            add_unavailable(r, "/runtime/qnn/soc_model",
                            "The QNN EP does not report the device's SoC model: its "
                            "compatibility info carries only its soc_model option (not set).");
        if (r.known.session && !r.runtime.qnn_backend_file_version)
            add_unavailable(r, "/runtime/qnn/backend_file_version",
                            "The version resource of the loaded QnnHtp.dll could not be read.");
    }
    if (!t.trials.empty() && !t.first_inference_ms) {
        add_unavailable(r, "/timing/first_inference_ms",
                        "There were no warm-up iterations to time separately.");
    }
    if (r.batch_size <= 0) {
        add_unavailable(r, "/config/batch_size",
                        "The batch size could not be determined: the run failed before the "
                        "model's input shape was known.");
    }
    // The ONNX Runtime and QNN modules, if no session snapshot was taken.
    if (r.known.runtime && r.runtime.loaded_modules.empty())
        r.runtime.loaded_modules = list_ort_modules();
}

std::string format_summary(const BenchmarkResult& r) {
    std::ostringstream out;
    const RunConfig& c = r.config;
    const ProviderDescription& p = r.runtime.provider;
    out << "Model       " << r.model.file.file_name << " (" << r.model.precision.stored_precision
        << ", " << r.model.file.size_bytes << " bytes, sha256 " << r.model.file.sha256.substr(0, 12)
        << ")\n";
    out << "Provider    " << p.execution_provider << ", " << c.threads
        << " intra-op threads, binding " << schema_name(c.binding) << ", graph optimization "
        << schema_name(c.graph_opt) << "\n";
    if (p.qnn) {
        out << "NPU         strict " << (strict_mode(c) ? "on" : "off") << ", compute precision "
            << r.compute_precision.value_or("-") << " (model " << r.model.precision.stored_precision
            << ")\n";
        out << "EP options ";
        for (const auto& [key, value] : p.ep_options)
            out << " " << key << "=" << value;
        out << "\nSession cfg";
        for (const auto& [key, value] : p.session_config)
            out << " " << key << "=" << value;
        out << "\n";
    }
    out << "Trials      " << c.trials << " x " << c.iterations << " measured iterations ("
        << c.warmup << " warm-up, " << c.cooldown_ms << " ms cooldown), batch size " << r.batch_size
        << "\n";
    out << "Input       " << schema_name(c.input) << "\n";
    if (r.batch_equivalence && r.batch_size > 1) {
        const BatchEquivalence& be = *r.batch_equivalence;
        out << "Batch       " << r.batch_size << " vs 1: ";
        if (!be.bitwise_equal_to_b1)
            out << "unknown (no reference data)";
        else if (*be.bitwise_equal_to_b1)
            out << "bit-identical per-item outputs (reference outputs)";
        else if (!be.predictions_equal_to_b1)
            out << "outputs differ, predictions not compared";
        else
            out << (*be.predictions_equal_to_b1 ? "outputs differ in the last bits, same "
                                                  "predictions"
                                                : "DIFFERENT predictions")
                << " (" << be.predictions_differing.value_or(0) << " of "
                << be.predictions_compared.value_or(0) << " differ, " << be.predictions_basis
                << ")";
        out << "\n";
    }

    const TimingResult& t = r.timing;
    const EpContextCacheResult& cache = r.ep_cache;
    if (cache.enabled()) {
        out << "EP cache    " << schema_name(cache.mode) << ", "
            << cache.state.value_or("not examined");
        if (cache.key_id)
            out << ", key " << cache.key_id->substr(0, 12);
        if (cache.verified_equal_outputs)
            out << (*cache.verified_equal_outputs ? ", outputs equal to a compiled session"
                                                  : ", OUTPUTS DIFFER from a compiled session");
        if (t.session_init.context_build_ms)
            out << ", built in " << fmt(*t.session_init.context_build_ms, 1) << " ms";
        out << "\n";
        if (cache.context_path)
            out << "            " << path_to_utf8(*cache.context_path) << "\n";
        if (!cache.deleted_stale.empty())
            out << "            deleted " << cache.deleted_stale.size()
                << " file(s) of stale entries for this model\n";
    }
    out << "Session     create ms ("
        << (t.session_init.ep_context_state == "cached" ? "from the EP context cache"
                                                        : "compiled from the model")
        << "):";
    for (double ms : t.session_init.timed_ms)
        out << " " << fmt(ms, 1);
    out << "  (median " << fmt(t.session_init.session_create_ms, 1) << ")\n";
    if (c.init_breakdown && !t.trials.empty() && t.trials.front().session_init.context_build_ms) {
        auto median_of = [&t](std::optional<double> TrialSessionInit::*member) {
            std::vector<double> v;
            for (const TrialResult& trial : t.trials)
                if (const std::optional<double>& x = trial.session_init.*member)
                    v.push_back(*x);
            std::sort(v.begin(), v.end());
            return v.empty() ? std::string("-") : fmt(nearest_rank(v, 50), 1);
        };
        out << "Init        median of " << t.trials.size() << " trials: cold "
            << median_of(&TrialSessionInit::cold_ms) << " ms, context build "
            << median_of(&TrialSessionInit::context_build_ms) << " ms, cached "
            << median_of(&TrialSessionInit::cached_init_ms) << " ms\n";
    }
    out << "First run   " << (t.first_inference_ms ? fmt(*t.first_inference_ms, 3) + " ms" : "-")
        << " (first warm-up iteration of trial 1)\n";
    if (p.qnn) {
        out << "QNN         sdk " << p.qnn->sdk_version.value_or("-") << " (build "
            << p.qnn->backend_build.value_or("-") << "), HTP arch " << p.qnn->htp_arch.value_or("-")
            << ", SoC " << p.qnn->soc_model.value_or("-") << "\n";
    }
    if (t.profiled_pass) {
        std::vector<double> acc = t.profiled_pass->accelerator_ms();
        std::sort(acc.begin(), acc.end());
        out << "Accelerator "
            << (acc.empty() ? std::string("-") : fmt(nearest_rank(acc, 50)) + " ms median")
            << " (profiled pass: " << acc.size()
            << " inferences in a separate profiled session; not a latency)\n";
    }

    if (r.system.power) {
        const PowerState& pw = *r.system.power;
        out << "Power       " << pw.source.value_or("source unknown") << ", mode "
            << pw.power_mode.value_or("unknown") << ", plan " << pw.power_plan.value_or("unknown")
            << ", battery saver "
            << (pw.battery_saver ? (*pw.battery_saver ? "on" : "off") : "unknown");
        if (pw.battery_percent)
            out << ", battery " << *pw.battery_percent << "%";
        out << "\n";
    }
    out << "CPU load    pre-run "
        << (r.system.load ? fmt(r.system.load->cpu_utilization_ratio * 100.0, 1) + "% over " +
                                fmt(r.system.load->sample_window_s, 1) + " s"
                          : std::string("-"));
    {
        std::optional<double> highest;
        int highest_trial = 0;
        for (const TrialResult& trial : t.trials) {
            if (trial.sequence > 0 && trial.idle_before &&
                (!highest || trial.idle_before->cpu_utilization_ratio > *highest)) {
                highest = trial.idle_before->cpu_utilization_ratio;
                highest_trial = trial.trial_index;
            }
        }
        if (highest) {
            out << ", highest cooldown " << fmt(*highest * 100.0, 1) << "% (before trial "
                << highest_trial + 1 << ")";
        }
        std::optional<double> during;
        int during_trial = 0;
        for (const TrialResult& trial : t.trials) {
            if (trial.during && (!during || trial.during->cpu_utilization_ratio > *during)) {
                during = trial.during->cpu_utilization_ratio;
                during_trial = trial.trial_index;
            }
        }
        if (during) {
            out << ", highest during trials " << fmt(*during * 100.0, 1) << "% (trial "
                << during_trial + 1 << ", own process excluded)";
        }
    }
    out << "\n";
    auto mib = [](double bytes) { return fmt(bytes / (1024.0 * 1024.0), 1) + " MiB"; };
    out << "Memory      working set "
        << mib(static_cast<double>(r.memory.working_set_after_last_trial_bytes))
        << " after the last trial, peak "
        << mib(static_cast<double>(r.memory.peak_working_set_bytes)) << ", session-create delta "
        << (r.memory.session_create_working_set_delta_bytes
                ? (*r.memory.session_create_working_set_delta_bytes >= 0 ? "+" : "") +
                      mib(static_cast<double>(*r.memory.session_create_working_set_delta_bytes)) +
                      " (median)"
                : std::string("-"))
        << " (process only)\n\n";

    char line[160];
    std::snprintf(line, sizeof(line), "%-10s %10s %10s %10s %10s %10s %10s %12s\n", "ms", "mean",
                  "p50", "p95", "p99", "min", "max", "inferences/s");
    out << line;
    auto row = [&](const std::string& name, const LatencyStats& s) {
        std::snprintf(line, sizeof(line), "%-10s %10s %10s %10s %10s %10s %10s %12s\n",
                      name.c_str(), fmt(s.mean_ms).c_str(), fmt(s.p50_ms).c_str(),
                      fmt_opt(s.p95_ms).c_str(), fmt_opt(s.p99_ms).c_str(), fmt(s.min_ms).c_str(),
                      fmt(s.max_ms).c_str(), fmt(s.inferences_per_sec, 1).c_str());
        out << line;
    };
    for (const TrialResult& trial : t.trials)
        row("trial " + std::to_string(trial.trial_index + 1), trial.stats);
    if (t.aggregate)
        row("aggregate", t.aggregate->stats);

    if (t.aggregate) {
        const TrialVariation& v = t.aggregate->trial_variation;
        out << "\nTrial means " << fmt(v.trial_mean_min_ms) << " .. " << fmt(v.trial_mean_max_ms)
            << " ms";
        if (v.trial_mean_cv_ratio)
            out << ", CV " << fmt(*v.trial_mean_cv_ratio * 100.0, 2) << "%";
        out << "\n";
    }
    const NodeAssignment& na = r.execution.node_assignment;
    out << "Placement   " << na.total_nodes << " nodes:";
    for (const auto& [ep, n] : na.nodes_by_ep)
        out << " " << ep << " " << n;
    if (na.qnn_validation)
        out << ", QNN partitions " << qnn_partitions(na);
    out << (r.execution.full_device_execution_verified ? " (all on the requested EP)"
                                                       : " (NOT all on the requested EP)")
        << "\n";
    if (const std::optional<QnnValidation>& qv = na.qnn_validation) {
        out << "QNN check  ";
        for (std::size_t i = 0; i < qv->passes.size(); ++i) {
            out << " pass " << i + 1 << ": " << qv->passes[i].passed_nodes << " passed, "
                << qv->passes[i].failed_nodes << " failed;";
        }
        out << " rejected: "
            << (qv->rejected_nodes.empty() ? "none" : describe_nodes(qv->rejected_nodes)) << "\n";
    }

    bool printed = false;
    for (const ResultWarning& w : r.warnings) {
        if (w.code == "value_unavailable")
            continue;  // expected until system state and memory are recorded
        if (!printed)
            out << "\nWarnings\n";
        printed = true;
        out << "  [" << w.code << "] " << w.message << "\n";
    }
    if (!r.notes.empty()) {
        out << "\nNotes\n";
        for (const ResultWarning& n : r.notes)
            out << "  [" << n.code << "] " << n.message << "\n";
    }
    return out.str();
}

}  // namespace edgebench
