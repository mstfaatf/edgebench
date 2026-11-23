// run_benchmark() failure paths in-process: interrupts raised at chosen points
// through the stop hook, and providers or sessions that fail or misbehave after
// the first trial. Every failed result must be written and validate.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "edgebench/benchmark.hpp"
#include "edgebench/interrupt.hpp"
#include "edgebench/ort_loader.hpp"
#include "edgebench/result_files.hpp"
#include "process_util.hpp"
#include "test_runtime.hpp"
#include "validator.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;
namespace eb = edgebench;
using json = nlohmann::ordered_json;
using eb::StopPoint;

namespace {

const fs::path kSourceDir = fs::path(EDGEBENCH_SOURCE_DIR).make_preferred();
const fs::path kFixtures = kSourceDir / "engine" / "tests" / "fixtures" / "models";

// Clears the process-wide stop flag before and after each test.
struct StopGuard {
    StopGuard() { eb::reset_stop(); }
    ~StopGuard() { eb::reset_stop(); }
};

eb::RunConfig short_run(int trials = 2) {
    eb::RunConfig c;
    c.model = kFixtures / "fixed_float32.onnx";
    c.trials = trials;
    c.warmup = 3;
    c.iterations = 10;
    c.cooldown_ms = 0;
    c.load_sample_ms = 0;
    return c;
}

std::unique_ptr<eb::Provider> real_provider(const eb::RunConfig& c) {
    auto p = eb::make_provider(eb::test::shared_runtime(), c);
    if (!p)
        throw std::runtime_error(p.error().message);
    return std::move(p).value();
}

// A hook that requests a stop when it reaches (point, trial, iteration).
eb::StopHook stop_at(StopPoint point, int trial, int iteration = 0) {
    return [=](StopPoint p, int t, int i) {
        if (p == point && t == trial && i == iteration)
            eb::request_stop("the test hook");
    };
}

bool has_warning(const eb::BenchmarkResult& r, std::string_view code) {
    return std::any_of(r.warnings.begin(), r.warnings.end(),
                       [&](const eb::ResultWarning& w) { return w.code == code; });
}

const eb::ResultWarning* find_warning(const eb::BenchmarkResult& r, std::string_view code) {
    for (const eb::ResultWarning& w : r.warnings) {
        if (w.code == code)
            return &w;
    }
    return nullptr;
}

// Writes the result and checks it with validate_results.py --strict; returns the document.
json write_and_validate(const eb::BenchmarkResult& r) {
    eb::test::TempDir dir;
    auto written = eb::write_result_files(r, dir.path(), "failure_path");
    EXPECT_TRUE(written) << (written ? "" : written.error().message);
    if (!written)
        return {};
    eb::test::validate_or_skip(written.value().result_file);
    return written.value().document;
}

void expect_interrupted(const eb::BenchmarkResult& r, std::string_view stage,
                        std::size_t completed_trials) {
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.status(), "failed");
    EXPECT_EQ(r.failure->error.kind, "interrupted");
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::interrupted);
    EXPECT_EQ(r.failure->stage, stage);
    EXPECT_EQ(r.timing.trials.size(), completed_trials);
    EXPECT_FALSE(r.known.placement);
    EXPECT_NE(r.failure->error.message.find("the test hook"), std::string::npos)
        << r.failure->error.message;
}

// Forwards to a real provider; can fail create_session on a given call, fail
// record_placement, or change the outputs of every session after the first.
class FaultyProvider final : public eb::Provider {
public:
    enum class Outputs { unchanged, same_top1, different_top1 };

    explicit FaultyProvider(const eb::RunConfig& c) : inner_(real_provider(c)) {}

    std::string_view name() const override { return inner_->name(); }
    std::string_view device_type() const override { return inner_->device_type(); }
    const eb::ProviderDescription& describe() const override { return inner_->describe(); }

    eb::Result<std::unique_ptr<eb::BenchSession>> create_session(const fs::path& model,
                                                                 const eb::RunConfig& c) override {
        const int call = sessions_++;
        if (call == fail_session_call) {
            return eb::Error{eb::ExitCode::ort_runtime_error, "ort_runtime_error",
                             "session creation failed on purpose"};
        }
        auto s = inner_->create_session(model, c);
        if (!s || call == 0 || outputs == Outputs::unchanged)
            return s;
        return std::unique_ptr<eb::BenchSession>(
            std::make_unique<ChangedOutputs>(std::move(s).value(), outputs));
    }

    eb::Result<eb::NodeAssignment> record_placement(const fs::path& model,
                                                    const eb::RunConfig& c) override {
        if (fail_placement) {
            return eb::Error{eb::ExitCode::ort_runtime_error, "ort_runtime_error",
                             "placement session failed on purpose"};
        }
        if (placement)
            return *placement;
        return inner_->record_placement(model, c);
    }

    int fail_session_call = -1;
    bool fail_placement = false;
    std::optional<eb::NodeAssignment> placement;  // reported instead of the real one
    Outputs outputs = Outputs::unchanged;

private:
    // Wraps a session and changes output 0 after every run.
    class ChangedOutputs final : public eb::BenchSession {
    public:
        ChangedOutputs(std::unique_ptr<eb::BenchSession> inner, Outputs mode)
            : inner_(std::move(inner)), mode_(mode) {}
        const std::vector<eb::TensorInfo>& inputs() const override { return inner_->inputs(); }
        const std::vector<eb::TensorInfo>& outputs() const override { return inner_->outputs(); }
        std::int64_t batch_size() const override { return inner_->batch_size(); }
        eb::RunBinding run_binding() const override { return inner_->run_binding(); }
        std::span<float> input_data(std::size_t i) override { return inner_->input_data(i); }
        double session_create_ms() const override { return inner_->session_create_ms(); }
        std::optional<std::int64_t> session_create_working_set_delta_bytes() const override {
            return inner_->session_create_working_set_delta_bytes();
        }
        eb::Result<void> run_once() override {
            eb::Result<void> r = inner_->run_once();
            if (!r)
                return r;
            const std::span<const float> y = inner_->output(0).as_float();
            changed_.assign(y.begin(), y.end());
            const auto top = static_cast<std::size_t>(
                std::max_element(changed_.begin(), changed_.end()) - changed_.begin());
            if (mode_ == Outputs::same_top1)
                changed_[top] += 0.5f;
            else
                changed_[(top + 1) % changed_.size()] = changed_[top] + 1.0f;
            return {};
        }
        eb::TensorView output(std::size_t i) const override {
            if (i != 0)
                return inner_->output(i);
            return {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::as_bytes(std::span(changed_))};
        }

    private:
        std::unique_ptr<eb::BenchSession> inner_;
        Outputs mode_;
        std::vector<float> changed_;
    };

    std::unique_ptr<eb::Provider> inner_;
    int sessions_ = 0;
};

eb::BenchmarkResult run(eb::Provider& p, const eb::RunConfig& c, const eb::StopHook& hook = {}) {
    return eb::run_benchmark(eb::test::shared_runtime(), p, c, {}, hook);
}

}  // namespace

// --------------------------------------------------------------------------- interrupts

TEST(Interrupt, BeforeTheFirstTrial) {
    StopGuard guard;
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::before_session, 0));
    expect_interrupted(r, "session_creation", 0);
    EXPECT_FALSE(has_warning(r, "interrupted_trial_discarded"));  // nothing was measured
    const json doc = write_and_validate(r);
    EXPECT_EQ(doc.at("error").at("code"), "interrupted");
    EXPECT_TRUE(doc.at("timing").is_null());
    EXPECT_TRUE(doc.at("raw").is_null());
}

TEST(Interrupt, MidSecondTrialDropsThePartialTrial) {
    StopGuard guard;
    const eb::RunConfig c = short_run(3);
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::measured, 1, 5));
    expect_interrupted(r, "measurement", 1);
    const eb::ResultWarning* w = find_warning(r, "interrupted_trial_discarded");
    ASSERT_NE(w, nullptr);
    EXPECT_NE(w->message.find("trial 2 of 3 was interrupted after 5 of 10 measured iterations"),
              std::string::npos)
        << w->message;
    EXPECT_NE(r.failure->error.message.find("trial 2 of 3, measured iteration 6"),
              std::string::npos)
        << r.failure->error.message;
    const json doc = write_and_validate(r);
    EXPECT_EQ(doc.at("timing").at("trials").size(), 1u);
    EXPECT_EQ(doc.at("timing").at("aggregate").at("stats").at("count"), 10);
    EXPECT_FALSE(doc.at("raw").is_null());
}

TEST(Interrupt, DuringWarmUp) {
    StopGuard guard;
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::warmup, 0, 1));
    expect_interrupted(r, "warmup", 0);
    EXPECT_TRUE(has_warning(r, "interrupted_trial_discarded"));
    write_and_validate(r);
}

TEST(Interrupt, AfterSessionCreation) {
    StopGuard guard;
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::after_session, 1));
    expect_interrupted(r, "session_creation", 1);
    write_and_validate(r);
}

TEST(Interrupt, DuringACooldownEndsItAtOnce) {
    StopGuard guard;
    eb::RunConfig c = short_run();
    c.cooldown_ms = 60'000;  // would take a minute without the interrupt
    FaultyProvider p(c);
    const auto t0 = std::chrono::steady_clock::now();
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::idle, 0));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(elapsed, std::chrono::seconds(20));
    expect_interrupted(r, "measurement", 0);
    EXPECT_FALSE(has_warning(r, "interrupted_trial_discarded"));
    write_and_validate(r);
}

TEST(Interrupt, FromAnotherThreadDuringACooldown) {
    StopGuard guard;
    eb::RunConfig c = short_run();
    c.cooldown_ms = 60'000;
    FaultyProvider p(c);
    std::thread stopper;
    const auto t0 = std::chrono::steady_clock::now();
    const eb::BenchmarkResult r = run(p, c, [&stopper](StopPoint point, int trial, int) {
        if (point == StopPoint::idle && trial == 0) {
            stopper = std::thread([] {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                eb::request_stop("the test hook");
            });
        }
    });
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    stopper.join();
    EXPECT_LT(elapsed, std::chrono::seconds(20));  // the wait ended on the event
    expect_interrupted(r, "measurement", 0);
}

TEST(Interrupt, BeforePlacementSkipsIt) {
    StopGuard guard;
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::before_placement, 0));
    expect_interrupted(r, "session_creation", 2);  // both trials kept
    const json doc = write_and_validate(r);
    EXPECT_EQ(doc.at("timing").at("trials").size(), 2u);
    // No placement: execution is unverified (null fields), never "verified".
    const json& execution = doc.at("execution");
    EXPECT_TRUE(execution.is_null() || execution.at("full_device_execution_verified").is_null())
        << execution.dump();
}

TEST(Interrupt, AfterPlacement) {
    StopGuard guard;
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c, stop_at(StopPoint::after_placement, 0));
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->error.kind, "interrupted");
    EXPECT_EQ(r.timing.trials.size(), 2u);
    write_and_validate(r);
}

// --------------------------------------------------------------------------- faults

TEST(FailurePaths, SessionCreationFailsOnTheSecondTrial) {
    const eb::RunConfig c = short_run(3);
    FaultyProvider p(c);
    p.fail_session_call = 1;
    const eb::BenchmarkResult r = run(p, c);
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->stage, "session_creation");
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::ort_runtime_error);
    EXPECT_NE(r.failure->error.message.find("trial 2 of 3"), std::string::npos)
        << r.failure->error.message;
    EXPECT_EQ(r.timing.trials.size(), 1u);
    write_and_validate(r);
}

TEST(FailurePaths, PlacementFailureFailsTheRun) {
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    p.fail_placement = true;
    const eb::BenchmarkResult r = run(p, c);
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->stage, "session_creation");
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::ort_runtime_error);
    EXPECT_EQ(r.timing.trials.size(), 2u);
    EXPECT_FALSE(r.known.placement);
    write_and_validate(r);
}

TEST(FailurePaths, OutputsDifferingWithTheSameTop1AreAWarning) {
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    p.outputs = FaultyProvider::Outputs::same_top1;
    const eb::BenchmarkResult r = run(p, c);
    EXPECT_FALSE(r.failure.has_value()) << r.failure->error.message;
    const eb::ResultWarning* w = find_warning(r, "nondeterministic_outputs");
    ASSERT_NE(w, nullptr);
    EXPECT_NE(w->message.find("output 'y' of trial 2"), std::string::npos) << w->message;
}

TEST(FailurePaths, OutputsWithADifferentTop1FailTheRun) {
    const eb::RunConfig c = short_run();
    FaultyProvider p(c);
    p.outputs = FaultyProvider::Outputs::different_top1;
    const eb::BenchmarkResult r = run(p, c);
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->error.kind, "inconsistent_outputs");
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::internal_error);
    EXPECT_EQ(r.failure->stage, "measurement");
    EXPECT_EQ(r.timing.trials.size(), 1u);
    write_and_validate(r);
}

TEST(FailurePaths, RunFailureDuringWarmUpAndMeasurement) {
    eb::RunConfig c = short_run();
    c.model = kFixtures / "runtime_error.onnx";
    {
        FaultyProvider p(c);
        const eb::BenchmarkResult r = run(p, c);
        ASSERT_TRUE(r.failure.has_value());
        EXPECT_EQ(r.failure->stage, "warmup");
        EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::ort_runtime_error);
        EXPECT_NE(r.failure->error.message.find("trial 1 of 2, warm-up iteration 1"),
                  std::string::npos)
            << r.failure->error.message;
        write_and_validate(r);
    }
    c.warmup = 0;
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c);
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->stage, "measurement");
    write_and_validate(r);
}

TEST(FailurePaths, ThreadsAboveLogicalProcessorsIsAWarning) {
    eb::RunConfig c = short_run(1);
    c.threads = 256;
    FaultyProvider p(c);
    const eb::BenchmarkResult r = run(p, c);
    EXPECT_FALSE(r.failure.has_value());
    const eb::ResultWarning* w = find_warning(r, "threads_exceed_logical_processors");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->path, std::optional<std::string>("/config/intra_op_threads"));
    EXPECT_NE(eb::format_summary(r).find("threads_exceed_logical_processors"), std::string::npos);
}

TEST(FailurePaths, NodesOffTheRequestedEpAreAWarning) {
    const eb::RunConfig c = short_run(1);
    FaultyProvider p(c);
    eb::NodeAssignment mixed;
    mixed.total_nodes = 3;
    mixed.nodes_by_ep = {{"CPUExecutionProvider", 2}, {"OtherExecutionProvider", 1}};
    p.placement = mixed;
    const eb::BenchmarkResult r = run(p, c);
    EXPECT_FALSE(r.failure.has_value());
    EXPECT_FALSE(r.execution.full_device_execution_verified);
    const eb::ResultWarning* w = find_warning(r, "nodes_not_on_requested_ep");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->message, "1 of 3 nodes are not on CPUExecutionProvider");
    EXPECT_NE(eb::format_summary(r).find("(NOT all on the requested EP)"), std::string::npos);
}

// --------------------------------------------------------------------------- foreign modules

namespace {

// Loads a copy of a system DLL under an ONNX Runtime-like name from a temporary
// directory, i.e. outside the bundle, for as long as it lives.
class ForeignModule {
public:
    ForeignModule() : dir_(L"edgebench_foreign_module") {
        path_ = dir_.path() / "onnxruntime_foreign_test.dll";
        fs::copy_file(LR"(C:\Windows\System32\version.dll)", path_);
        module_ = LoadLibraryExW(path_.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    ~ForeignModule() {
        if (module_)
            FreeLibrary(module_);
    }
    ForeignModule(const ForeignModule&) = delete;
    ForeignModule& operator=(const ForeignModule&) = delete;
    bool loaded() const { return module_ != nullptr; }
    const fs::path& path() const { return path_; }

private:
    eb::test::TempDir dir_;  // destroyed after the module is freed
    fs::path path_;
    HMODULE module_ = nullptr;
};

}  // namespace

TEST(FailurePaths, ModuleOutsideTheBundleStopsTheRun) {
    const eb::RunConfig c = short_run(2);
    const std::unique_ptr<eb::Provider> p = real_provider(c);
    eb::BenchmarkResult r;
    std::string resolved;
    {
        ForeignModule foreign;
        ASSERT_TRUE(foreign.loaded());
        resolved = eb::path_to_utf8(eb::resolve_path(foreign.path()));
        r = run(*p, c);
    }
    ASSERT_TRUE(r.failure.has_value());
    EXPECT_EQ(r.failure->error.exit_code, eb::ExitCode::ort_load_failure);
    EXPECT_EQ(r.failure->error.kind, "module_outside_bundle");
    EXPECT_EQ(r.failure->stage, "session_creation");
    const std::string& m = r.failure->error.message;
    EXPECT_NE(m.find("trial 1 of 2"), std::string::npos) << m;
    EXPECT_NE(m.find("onnxruntime_foreign_test.dll (" + resolved + ")"), std::string::npos) << m;
    EXPECT_TRUE(r.timing.trials.empty());
    EXPECT_FALSE(r.known.placement);
    EXPECT_TRUE(std::any_of(
        r.runtime.loaded_modules.begin(), r.runtime.loaded_modules.end(),
        [](const eb::LoadedModule& lm) { return lm.name == "onnxruntime_foreign_test.dll"; }));
    const json doc = write_and_validate(r);
    ASSERT_FALSE(doc.empty());
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("error").at("code"), "module_outside_bundle");
    EXPECT_EQ(doc.at("error").at("stage"), "session_creation");
}

// --------------------------------------------------------------------------- warnings from trial
// data

namespace {

eb::TrialResult fake_trial(int index, double latency_ms) {
    eb::TrialResult t;
    t.trial_index = index;
    t.sequence = index;
    t.session_create_ms = 10.0;
    t.latencies_ms.assign(20, latency_ms);
    t.stats = eb::compute_stats(t.latencies_ms, 1);
    return t;
}

eb::BenchmarkResult result_with_trials(std::vector<eb::TrialResult> trials) {
    eb::BenchmarkResult r;
    r.config = short_run(static_cast<int>(trials.size()));
    r.batch_size = 1;
    for (const eb::TrialResult& t : trials)
        r.timing.session_init.cold_ms.push_back(t.session_create_ms);
    r.timing.trials = std::move(trials);
    return r;
}

}  // namespace

TEST(FailurePaths, TrialSpreadIsAWarningInTheResultAndSummary) {
    eb::BenchmarkResult r = result_with_trials({fake_trial(0, 3.8), fake_trial(1, 9.6)});
    eb::finish_result(r);
    const eb::ResultWarning* w = find_warning(r, "trial_spread_high");
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->path, std::optional<std::string>("/timing/aggregate/trial_variation"));
    EXPECT_NE(w->message.find("3.8000 to 9.6000 ms"), std::string::npos) << w->message;
    EXPECT_NE(eb::format_summary(r).find("[trial_spread_high]"), std::string::npos);

    eb::BenchmarkResult steady = result_with_trials({fake_trial(0, 3.8), fake_trial(1, 3.9)});
    eb::finish_result(steady);
    EXPECT_FALSE(has_warning(steady, "trial_spread_high"));
}

TEST(FailurePaths, BackgroundLoadDuringATrialIsAWarning) {
    std::vector<eb::TrialResult> trials{fake_trial(0, 4.0), fake_trial(1, 4.0)};
    trials[0].during = eb::LoadSample{0.02, 0.8};
    trials[1].during = eb::LoadSample{0.18, 0.8};
    eb::BenchmarkResult r = result_with_trials(std::move(trials));
    eb::finish_result(r);
    const eb::ResultWarning* w = nullptr;
    for (const eb::ResultWarning& x : r.warnings) {
        if (x.code == "background_cpu_load_high" && x.message.find("during") != std::string::npos)
            w = &x;
    }
    ASSERT_NE(w, nullptr);
    EXPECT_NE(w->message.find("18.0% during trial 2 of 2"), std::string::npos) << w->message;
    EXPECT_NE(eb::format_summary(r).find("highest during trials 18.0% (trial 2"), std::string::npos)
        << eb::format_summary(r);

    std::vector<eb::TrialResult> quiet{fake_trial(0, 4.0)};
    quiet[0].during = eb::LoadSample{0.10, 0.8};  // not above the threshold
    eb::BenchmarkResult q = result_with_trials(std::move(quiet));
    eb::finish_result(q);
    for (const eb::ResultWarning& x : q.warnings) {
        if (x.code == "background_cpu_load_high")
            ADD_FAILURE() << x.message;
    }
}
