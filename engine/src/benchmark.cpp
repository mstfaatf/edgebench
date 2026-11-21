#include "edgebench/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

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

// One trial: fresh session, warm-up, measured iterations. Returns the session's
// final outputs through `final_outputs`.
Result<TrialResult> run_trial(Provider& provider, const RunConfig& config, const ScheduledTrial& st,
                              BenchmarkResult& result,
                              std::vector<std::vector<std::byte>>& final_outputs) {
    const std::string where = trial_label(st.trial_index, config.trials);
    Result<std::unique_ptr<BenchSession>> created =
        provider.create_session(result.model.file.path, config);
    if (!created)
        return with_context(created.error(), where);
    std::unique_ptr<BenchSession> session = std::move(created).value();

    TrialResult trial;
    trial.trial_index = st.trial_index;
    trial.sequence = st.sequence;
    trial.session_create_ms = session->session_create_ms();

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

    for (int i = 0; i < config.warmup; ++i) {
        const Clock::time_point t0 = Clock::now();
        if (Result<void> r = run("warm-up", i); !r)
            return r.error();
        if (i == 0)
            trial.first_inference_ms = ms_between(t0, Clock::now());
    }

    trial.latencies_ms.reserve(static_cast<std::size_t>(config.iterations));
    for (int i = 0; i < config.iterations; ++i) {
        const Clock::time_point t0 = Clock::now();
        const Result<void> r = session->run_once();
        const Clock::time_point t1 = Clock::now();
        if (!r) {
            return with_context(r.error(), where + ", measured iteration " + std::to_string(i + 1));
        }
        trial.latencies_ms.push_back(ms_between(t0, t1));
    }

    for (std::size_t i = 0; i < session->outputs().size(); ++i) {
        if (const auto bad = first_non_finite(session->output(i))) {
            return Error{ExitCode::internal_error, "non_finite_output",
                         where + ": output '" + session->outputs()[i].name +
                             "' has a non-finite value (NaN or infinity) at element " +
                             std::to_string(*bad)};
        }
    }
    final_outputs = copy_outputs(*session);
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

std::string generated_input_description(const RunConfig& config) {
    const std::string data = config.input == InputKind::random
                                 ? "random values in [0, 1) from SplitMix64 (seed " +
                                       std::to_string(config.effective_seed()) + ")"
                                 : "the constant " + fmt(config.effective_constant_value(), 6);
    return "No preprocessing: inputs are generated (" + data +
           ") once per session, before timing, and reused for every iteration.";
}

Result<BenchmarkResult> run_benchmark(Runtime& runtime, Provider& provider, const RunConfig& config,
                                      const ProgressFn& progress) {
    auto report = [&progress](const std::string& message) {
        if (progress)
            progress(message);
    };

    BenchmarkResult r;
    r.schema_version = std::string(results_schema_version());
    r.run_id = new_run_id();
    r.timestamp_utc = utc_timestamp(std::chrono::system_clock::now());
    r.label = config.label;
    r.tool_version = std::string(version());
    r.tool = build_info();
    r.config = config;
    r.input_preprocessing = generated_input_description(config);
    r.timing.timer = steady_clock_info();

    // Model file facts first: reading the file also warms the file cache equally
    // for every trial.
    Result<ModelFileInfo> file = read_model_file_info(config.model);
    if (!file)
        return file.error();
    r.model.file = std::move(file).value();
    r.model.name = path_to_utf8(r.model.file.path.stem());
    const Result<OnnxGraphSummary> graph = inspect_onnx_file(r.model.file.path);

    r.system = query_system_info();
    r.runtime.onnxruntime = runtime.load_result();
    r.runtime.provider = provider.describe();
    try {
        r.runtime.ep_devices = runtime.ep_devices();
    } catch (const Ort::Exception& e) {
        return ort_error("ep_enumeration_failed", "enumerating EP devices", e.GetOrtErrorCode(),
                         e.what());
    }

    std::vector<std::vector<std::byte>> reference_outputs;
    for (const ScheduledTrial& st : schedule_trials(1, config.trials)) {
        if (st.sequence > 0 && config.cooldown_ms > 0) {
            report("cooldown " + std::to_string(config.cooldown_ms) + " ms");
            std::this_thread::sleep_for(std::chrono::milliseconds(config.cooldown_ms));
        }
        report(trial_label(st.trial_index, config.trials) + ": creating session, " +
               std::to_string(config.warmup) + " warm-up + " + std::to_string(config.iterations) +
               " measured iterations");
        std::vector<std::vector<std::byte>> outputs;
        Result<TrialResult> trial = run_trial(provider, config, st, r, outputs);
        if (!trial)
            return trial.error();
        if (st.sequence == 0) {
            // ONNX Runtime accepted the model; the file decoder must have too.
            if (!graph) {
                return Error{ExitCode::internal_error, graph.error().kind,
                             "cannot read the model's metadata: " + graph.error().message};
            }
            reference_outputs = std::move(outputs);
        } else if (Result<void> d =
                       check_determinism(r, st.trial_index, reference_outputs, outputs, r.warnings);
                   !d) {
            return d.error();
        }
        r.timing.session_init.cold_ms.push_back(trial.value().session_create_ms);
        r.timing.trials.push_back(std::move(trial).value());
    }

    // Placement last, in its own verbose session, so it cannot warm a timed session.
    report("recording node placement");
    Result<NodeAssignment> placement = provider.record_placement(r.model.file.path, config);
    if (!placement)
        return placement.error();
    const std::string requested_ep(execution_provider_name(config.provider));
    r.execution.node_assignment = std::move(placement).value();
    r.execution.full_device_execution_verified = r.execution.node_assignment.all_on(requested_ep);
    r.execution.verification_details =
        "Node placement from ONNX Runtime's verbose session log, recorded in a separate session "
        "with the same options after the timed trials.";
    if (!r.execution.full_device_execution_verified) {
        const NodeAssignment& p = r.execution.node_assignment;
        r.warnings.push_back({"nodes_not_on_requested_ep",
                              std::to_string(p.nodes_not_on(requested_ep)) + " of " +
                                  std::to_string(p.total_nodes) + " nodes are not on " +
                                  requested_ep,
                              std::string("/execution/node_assignment")});
    }

    // Model metadata from the file.
    const OnnxGraphSummary& g = graph.value();
    r.model.opset = g.opset;
    r.model.ir_version = g.ir_version;
    r.model.precision = infer_precision(g);
    if (!g.producer_name.empty())
        r.model.export_tool =
            g.producer_name + (g.producer_version.empty() ? "" : " " + g.producer_version);

    // Statistics.
    TimingResult& t = r.timing;
    t.first_inference_ms = t.trials.front().first_inference_ms;
    t.session_init.session_create_ms = [&] {
        std::vector<double> sorted = t.session_init.cold_ms;
        std::sort(sorted.begin(), sorted.end());
        return nearest_rank(sorted, 50);
    }();
    std::vector<double> pooled, means;
    for (const TrialResult& trial : t.trials) {
        pooled.insert(pooled.end(), trial.latencies_ms.begin(), trial.latencies_ms.end());
        means.push_back(trial.stats.mean_ms);
    }
    t.aggregate =
        AggregateResult{compute_stats(pooled, r.batch_size), compute_trial_variation(means)};

    // Values this build does not measure yet.
    add_unavailable(r, "/system/power/source", "The power source is not recorded yet.");
    add_unavailable(r, "/system/power/power_mode", "The Windows power mode is not recorded yet.");
    add_unavailable(r, "/system/load", "Background CPU load is not sampled yet.");
    add_unavailable(r, "/memory", "Process memory is not recorded yet.");
    if (r.runtime.provider.ep_device && !r.runtime.provider.ep_device->driver_version) {
        add_unavailable(r, "/runtime/ep_device/driver_version",
                        "The device driver version could not be determined.");
    }
    if (!t.first_inference_ms) {
        add_unavailable(r, "/timing/first_inference_ms",
                        "There were no warm-up iterations to time separately.");
    }
    return r;
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
        << " (first warm-up iteration of trial 1)\n\n";

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
