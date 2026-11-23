#include "edgebench/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

#include "edgebench/interrupt.hpp"
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
// the onnxruntime.dll that was loaded (the bundle). Checked after each session is
// created, when execution providers have pulled in their libraries. On failure
// the full module list goes into the result, so the foreign module is recorded.
Result<void> check_loaded_modules(BenchmarkResult& result) {
    const std::filesystem::path& dll = result.runtime.onnxruntime.dll_path;
    const std::filesystem::path bundle = dll.empty() ? executable_dir() : dll.parent_path();
    std::vector<LoadedModule> modules = list_ort_modules();
    const std::vector<LoadedModule> outside = find_modules_outside(modules, bundle);
    if (outside.empty())
        return {};
    std::string names;
    for (const LoadedModule& m : outside)
        names += (names.empty() ? "" : ", ") + m.name + " (" + path_to_utf8(m.path) + ")";
    result.runtime.loaded_modules = std::move(modules);
    return Error{ExitCode::ort_load_failure, "module_outside_bundle",
                 "ONNX Runtime/QNN module loaded from outside the bundle directory " +
                     path_to_utf8(bundle) + ": " + names};
}

// One trial: fresh session, warm-up, measured iterations. Stop requests are
// honoured before and after session creation and between iterations (never
// inside the timed bracket).
Result<TrialResult> run_trial(Provider& provider, const RunConfig& config, const ScheduledTrial& st,
                              BenchmarkResult& result, const StopHook& hook, TrialOutcome& out) {
    const std::string where = trial_label(st.trial_index, config.trials);
    out.failed_stage = "session_creation";
    if (should_stop(hook, StopPoint::before_session, st.trial_index, 0))
        return interrupted_error(where + ", before session creation");
    Result<std::unique_ptr<BenchSession>> created =
        provider.create_session(result.model.file.path, config);
    if (!created)
        return with_context(created.error(), where);
    std::unique_ptr<BenchSession> session = std::move(created).value();
    result.known.session = true;
    if (Result<void> modules = check_loaded_modules(result); !modules)
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
    // Background load across the measured iterations, read outside the timed brackets.
    const CpuTimes system_start = query_cpu_times();
    const std::optional<std::uint64_t> own_start = query_process_cpu_time();
    const Clock::time_point measured_start = Clock::now();
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
    {
        const Clock::time_point measured_end = Clock::now();
        const CpuTimes system_end = query_cpu_times();
        const std::optional<std::uint64_t> own_end = query_process_cpu_time();
        if (own_start && own_end) {
            if (const std::optional<double> background =
                    background_cpu_utilization(system_start, system_end, *own_start, *own_end)) {
                trial.during = LoadSample{
                    *background,
                    std::chrono::duration<double>(measured_end - measured_start).count()};
            }
        }
    }

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

    std::vector<std::vector<std::byte>> reference_outputs;
    for (const ScheduledTrial& st : schedule_trials(1, config.trials)) {
        // Every trial, the first included, starts after the same idle period, so
        // session creation always meets the CPU in the same power state.
        std::optional<LoadSample> idle;
        const std::string label = trial_label(st.trial_index, config.trials);
        if (const std::chrono::milliseconds wait = idle_before_trial(config, st.sequence);
            wait.count() > 0) {
            report("idle " + std::to_string(wait.count()) + " ms before " + label +
                   " (sampling background CPU load)");
            // Waits on the stop event, so an interrupt ends the idle period at once.
            const CpuTimes start = query_cpu_times();
            const auto t0 = Clock::now();
            const bool hook_stop = should_stop(hook, StopPoint::idle, st.trial_index, 0);
            const bool cut = hook_stop || wait_or_stop(wait);
            idle = LoadSample{cpu_utilization(start, query_cpu_times()).value_or(0.0),
                              std::chrono::duration<double>(Clock::now() - t0).count()};
            if (st.sequence == 0 && !cut)
                r.system.load = idle;
            if (cut) {
                set_failure(r, interrupted_error("idle period before " + label), "measurement");
                break;
            }
        }
        report(trial_label(st.trial_index, config.trials) + ": creating session, " +
               std::to_string(config.warmup) + " warm-up + " + std::to_string(config.iterations) +
               " measured iterations");
        TrialOutcome outcome;
        Result<TrialResult> trial = run_trial(provider, config, st, r, hook, outcome);
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
        trial.value().idle_before = idle;
        r.timing.session_init.cold_ms.push_back(trial.value().session_create_ms);
        r.timing.trials.push_back(std::move(trial).value());
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
            r.execution.full_device_execution_verified =
                r.execution.node_assignment.all_on(requested_ep);
            r.execution.verification_details =
                "Node placement from ONNX Runtime's verbose session log, recorded in a separate "
                "session with the same options after the timed trials.";
            if (!r.execution.full_device_execution_verified) {
                const NodeAssignment& p = r.execution.node_assignment;
                r.warnings.push_back({"nodes_not_on_requested_ep",
                                      std::to_string(p.nodes_not_on(requested_ep)) + " of " +
                                          std::to_string(p.total_nodes) + " nodes are not on " +
                                          requested_ep,
                                      std::string("/execution/node_assignment")});
            }
            // The placement session itself is not interruptible; stop right after it.
            if (should_stop(hook, StopPoint::after_placement, 0, 0))
                set_failure(r, interrupted_error("after node placement"), "session_creation");
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

    // Background load: the highest idle sample.
    {
        double highest = -1.0;
        std::string when;
        for (const TrialResult& trial : r.timing.trials) {
            if (trial.idle_before && trial.idle_before->cpu_utilization_ratio > highest) {
                highest = trial.idle_before->cpu_utilization_ratio;
                when = "before " + trial_label(trial.trial_index, config.trials);
            }
        }
        if (highest > kHighBackgroundLoad) {
            r.warnings.push_back({"background_cpu_load_high",
                                  "Background CPU load reached " + fmt(highest * 100.0, 1) + "% (" +
                                      when +
                                      "); other processes may have distorted the measurement.",
                                  std::string("/system/load")});
        }
    }
    // Background load during the measured iterations (this process's own CPU time
    // excluded): reported separately, because it affects the latencies directly.
    {
        const TrialResult* worst = nullptr;
        for (const TrialResult& trial : r.timing.trials) {
            if (trial.during && (!worst || trial.during->cpu_utilization_ratio >
                                               worst->during->cpu_utilization_ratio))
                worst = &trial;
        }
        if (worst && worst->during->cpu_utilization_ratio > kHighBackgroundLoad) {
            r.warnings.push_back(
                {"background_cpu_load_high",
                 "Background CPU load from other processes reached " +
                     fmt(worst->during->cpu_utilization_ratio * 100.0, 1) + "% during " +
                     trial_label(worst->trial_index, config.trials) +
                     " (this process's own CPU time excluded); other processes may have "
                     "distorted the measurement.",
                 std::nullopt});
        }
    }

    // Statistics of the completed trials.
    TimingResult& t = r.timing;
    if (!t.trials.empty()) {
        t.first_inference_ms = t.trials.front().first_inference_ms;
        std::vector<double> sorted = t.session_init.cold_ms;
        std::sort(sorted.begin(), sorted.end());
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
    out << "Trials      " << c.trials << " x " << c.iterations << " measured iterations ("
        << c.warmup << " warm-up, " << c.cooldown_ms << " ms cooldown), batch size " << r.batch_size
        << "\n";
    out << "Input       " << schema_name(c.input) << "\n";

    const TimingResult& t = r.timing;
    out << "Session     create ms:";
    for (double ms : t.session_init.cold_ms)
        out << " " << fmt(ms, 1);
    out << "  (median " << fmt(t.session_init.session_create_ms, 1) << ")\n";
    out << "First run   " << (t.first_inference_ms ? fmt(*t.first_inference_ms, 3) + " ms" : "-")
        << " (first warm-up iteration of trial 1)\n";

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
    out << (r.execution.full_device_execution_verified ? " (all on the requested EP)"
                                                       : " (NOT all on the requested EP)")
        << "\n";

    bool printed = false;
    for (const ResultWarning& w : r.warnings) {
        if (w.code == "value_unavailable")
            continue;  // expected until system state and memory are recorded
        if (!printed)
            out << "\nWarnings\n";
        printed = true;
        out << "  [" << w.code << "] " << w.message << "\n";
    }
    return out.str();
}

}  // namespace edgebench
