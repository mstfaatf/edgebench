// `edgebench suite`: runs a suite plan (suite_config.hpp) and keeps its manifest.
//
// Every planned run is a child process of the same executable (`edgebench run`
// with the hidden --suite-* options): a fresh process gives every run a real cold
// start, keeps the QNN compatibility query before the first QNN session of the
// process, and isolates ONNX Runtime teardown. Before the first run the suite
// runs its pre-flight gate (an ordered list of checks); then, per run, it idles
// for the configuration's cooldown, launches the run with its output in its own
// directory and its stdout and stderr in a log file, records the exit code, and
// validates the result with scripts/validate_results.py.
//
// The suite directory holds suite-manifest.json (schemas/suite-manifest.v1.0.0),
// rewritten atomically after every change of state, so an interrupted suite can
// be resumed: finished runs whose result still validates are kept, everything
// else runs again.
//
//   <suite dir>/suite-manifest.json
//   <suite dir>/suite-config.json          the configuration file as it was read
//   <suite dir>/preflight/doctor-<n>.json  one doctor report per pre-flight attempt
//   <suite dir>/runs/<short id>/           output directory of a run (results, run.log)
//   <suite dir>/runs/<short id>/cache_build/  the cache-building run of init "cached"
//   <suite dir>/qnn-context/               the suite's EP context cache
//   <suite dir>/discarded/<short id>-<n>/  output of an interrupted or failed attempt
#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "edgebench/result.hpp"
#include "edgebench/suite_config.hpp"

namespace edgebench::suite {

inline constexpr std::string_view kManifestVersion = "1.0.0";
inline constexpr std::string_view kManifestFileName = "suite-manifest.json";
inline constexpr std::string_view kConfigCopyName = "suite-config.json";

enum class RunStatus { pending, running, done, failed, skipped };
std::string_view to_string(RunStatus status);
std::optional<RunStatus> parse_run_status(std::string_view text);

// ------------------------------------------------------------------- pre-flight

enum class GateStatus { pass, warn, fail };
std::string_view to_string(GateStatus status);

/// The outcome of one pre-flight check.
struct GateResult {
    std::string name;
    GateStatus status = GateStatus::fail;
    std::string summary;
    nlohmann::ordered_json details = nlohmann::ordered_json::object();
};

/// What a pre-flight check may use.
struct PreflightContext {
    std::filesystem::path exe;        // edgebench.exe
    std::filesystem::path suite_dir;  // where reports go
    int attempt = 1;                  // 1-based: the n-th pre-flight of this suite
};

/// One check of the pre-flight gate. Checks run in order; any fail stops the suite
/// before its first run (and a warn too with preflight.fail_on_warn).
class PreflightCheck {
public:
    virtual ~PreflightCheck() = default;
    virtual std::string name() const = 0;
    virtual GateResult run(const PreflightContext& context) = 0;
};

/// `edgebench doctor --json --quick` as a child process; the report is saved as
/// preflight/doctor-<attempt>.json. fail: doctor exited non-zero or printed no
/// report; warn: doctor passed with warnings; details hold the per-check statuses
/// and the versions doctor reported.
std::unique_ptr<PreflightCheck> make_doctor_check();

/// The checks named in the configuration, in order (unknown names: none; the
/// configuration loader only accepts known names).
std::vector<std::unique_ptr<PreflightCheck>> make_preflight_checks(
    const std::vector<std::string>& names);

/// The checks a suite must pass for its results to be publishable.
const std::vector<std::string>& publish_gate_checks();

/// The gate's verdict over its checks: fail if any check failed, or warned with
/// `fail_on_warn`.
bool gate_passed(const std::vector<GateResult>& results, bool fail_on_warn);

// -------------------------------------------------------------------- estimate

/// Timing facts of one configuration, from an earlier result or a heuristic.
struct TimingBasis {
    double iteration_ms = 0;   // mean latency of one inference
    double cold_init_s = 0;    // session creation from the model
    double cached_init_s = 0;  // session creation from an EP context (NPU)
    double build_init_s = 0;   // session creation that writes an EP context (NPU)
    bool from_prior = false;   // measured (a matching earlier result) or heuristic
    std::string source;        // the result file, or "heuristic"
};

/// Earlier results to estimate from: every successful result document under
/// `roots`, keyed by model SHA256, provider, compute precision and batch size.
class PriorResults {
public:
    static PriorResults scan(const std::vector<std::filesystem::path>& roots);
    void add(const nlohmann::json& result, const std::string& source);
    std::optional<TimingBasis> find(const ResolvedConfiguration& c) const;
    std::size_t size() const { return entries_.size(); }

private:
    struct Entry {
        std::string key;
        TimingBasis basis;
    };
    std::vector<Entry> entries_;
};

/// A conservative guess when no earlier result matches (docs/suite.md).
TimingBasis heuristic_basis(const ResolvedConfiguration& c);

/// Seconds one run of `c` takes, cooldown before it included, given `basis`
/// (with a margin; it errs toward too long).
double estimate_run_s(const ResolvedConfiguration& c, const TimingBasis& basis);

/// Fixed costs of a run's process (start, ONNX Runtime load, model hash, placement
/// session bookkeeping) and of validating its result.
inline constexpr double kProcessOverheadS = 3.0;
inline constexpr double kNpuProcessOverheadS = 4.0;
inline constexpr double kValidationS = 1.5;
inline constexpr double kPreflightS = 5.0;
/// Every estimate is multiplied by this.
inline constexpr double kEstimateMargin = 1.25;

struct SuiteEstimate {
    double total_s = 0;
    std::vector<double> run_s;  // per planned run, in order
    int from_prior = 0;         // configurations estimated from earlier results
    int from_heuristic = 0;
};

SuiteEstimate estimate_suite(const SuitePlan& plan, const PriorResults& prior);

// --------------------------------------------------------------------- manifest

/// One planned run in the manifest.
struct RunRecord {
    std::string run_id;
    std::string configuration_id;
    std::string label;
    int order_index = 0;
    int repetition = 0;
    nlohmann::ordered_json configuration;  // ResolvedConfiguration::resolved
    RunStatus status = RunStatus::pending;
    int attempts = 0;
    std::optional<int> exit_code;
    std::optional<std::string> exit_code_name;
    std::optional<std::string> started_utc, ended_utc;
    std::optional<double> duration_s;
    std::optional<double> cooldown_s_before;
    double estimate_s = 0;
    std::optional<std::string> output_dir;   // relative to the suite directory
    std::optional<std::string> result_file;  // relative to the suite directory
    std::optional<std::string> result_sha256;
    std::optional<std::string> log_file;  // relative to the suite directory
    std::optional<std::string> failure;   // why it failed or was skipped
    /// validate_results.py: passed | failed | null (not validated).
    std::optional<std::string> validation;
    std::vector<std::string> validation_issues;
    std::vector<std::string> warnings;  // codes of the result's warnings (no value_unavailable)
    nlohmann::ordered_json cache_build = nullptr;  // init "cached": the build run
    std::vector<std::string> discarded;            // output of earlier attempts, moved aside
};

struct PreflightAttempt {
    std::string started_utc;
    bool skipped = false;
    bool passed = false;
    std::vector<GateResult> checks;
};

struct Invocation {
    std::string started_utc;
    std::optional<std::string> ended_utc;
    bool resumed = false;
    bool fail_fast = false;
    bool skip_preflight = false;
    std::optional<int> exit_code;
};

struct SuiteManifest {
    std::string suite_name;
    std::string config_file;  // absolute path as given at the start
    std::string config_sha256;
    nlohmann::json config;
    std::string created_utc;
    std::string updated_utc;
    std::string status = "planned";  // planned | running | interrupted | completed | aborted
    nlohmann::ordered_json tool;     // name, version, git_commit, git_dirty, build_type
    nlohmann::ordered_json versions = nullptr;  // from the latest doctor report
    std::string rotation;
    int repetitions = 1;
    std::string python;  // interpreter that runs the validator
    std::string validator;
    std::vector<PreflightAttempt> preflight;
    bool preflight_ever_skipped = false;
    std::vector<Invocation> invocations;
    SuiteEstimate estimate;
    std::vector<RunRecord> runs;

    bool publishable() const;
    std::vector<std::string> not_publishable_reasons() const;
};

nlohmann::ordered_json to_json(const SuiteManifest& m);
/// Reads a manifest back; usage_error naming the problem if it is not one.
Result<SuiteManifest> manifest_from_json(const nlohmann::json& j);

/// What --resume does with each run of a manifest.
enum class ResumeAction {
    keep,     // done and its result still validates
    run,      // pending or skipped: run it
    rerun,    // failed, or done but its result is missing or no longer validates
    discard,  // running: interrupted mid-run; its partial output is set aside, then run
};

/// The decision for one run. `result_valid` says whether a done run's result file
/// exists and validates (only asked for done runs).
ResumeAction resume_action(const RunRecord& run,
                           const std::function<bool(const RunRecord&)>& result_valid);

// ----------------------------------------------------------------------- running

struct SuiteOptions {
    std::filesystem::path exe;  // edgebench.exe to launch runs and doctor with
    std::optional<std::filesystem::path> config;
    std::optional<std::filesystem::path> resume;
    std::optional<std::filesystem::path> output_dir;
    std::optional<std::filesystem::path> python;
    bool dry_run = false;
    bool fail_fast = false;
    bool skip_preflight = false;
};

/// Test hooks, called on the suite's thread.
struct SuiteHooks {
    /// After a run's child process has started.
    std::function<void(const RunRecord&)> run_started;
    /// After a run has finished and the manifest was written.
    std::function<void(const RunRecord&)> run_finished;
};

/// The plan as a table plus the estimate (what --dry-run prints).
std::string format_plan(const SuitePlan& plan, const SuiteEstimate& estimate);

/// The summary printed when the suite ends.
std::string format_summary(const SuiteManifest& manifest, double wall_s);

/// The repository directory that holds scripts/validate_results.py: searched
/// upwards from each of `starts` (the suite uses the configuration file's
/// directory, the working directory and the executable's directory); nullopt if none.
std::optional<std::filesystem::path> find_repository(
    const std::vector<std::filesystem::path>& starts);

/// Runs `edgebench suite`. Returns the exit code: 0 every run done; 1 a run failed
/// or the pre-flight gate failed; 2 usage or configuration errors (also a resume
/// whose configuration changed); 6 the suite directory or manifest could not be
/// written; 130 interrupted.
int run_suite(const SuiteOptions& options, const SuiteHooks& hooks = {});

}  // namespace edgebench::suite
