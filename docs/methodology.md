# Measurement methodology

How EdgeBench measures inference latency, and what its numbers do and do not
mean. The evidence behind each choice is in the
[feasibility note](feasibility.md); the exact field definitions are in the
[results schema](results-schema.md), and the command behaviour in
[cli.md](cli.md#what-run-measures).

## What is timed

**Model execution only.** Each measured iteration times one inference call
(`Session::Run`, or the IoBinding run) with `std::chrono::steady_clock`, and
nothing else. Image decoding, resizing and normalization are not part of a run:
inputs are generated once per session from a seeded generator (or a constant)
before the first iteration. Every result records this as
`timing.scope = "model_execution"` together with a description of the input.
Preprocessing cost is deliberately excluded because it depends on the
application, not on the execution provider; for MobileNetV2, normalizing one
image in NumPy took 0.027 ms ([feasibility, section 7.2](feasibility.md#72-ai-hub-comparison)).

Session creation is measured separately (`timing.session_init.cold_ms`, one
value per trial) and is never mixed into the iteration latencies.

## Trial structure

A run is a number of **trials**. Every trial:

1. idles for the same period (`--cooldown-ms`, default 2 s) while sampling
   background CPU load; the first trial idles too (with `--cooldown-ms 0`, only
   the first trial idles, for `--load-sample-ms`);
2. creates a **fresh session**;
3. runs warm-up iterations (the first one is timed and reported separately as
   `first_inference_ms`, the rest are untimed);
4. runs the measured iterations, each timed on its own;
5. checks the outputs (finite, same top-1 as trial 1) and releases the session.

Fresh sessions make each trial independent of the previous one's caches and
allocations. The equal idle before every trial matters because session creation
and the first iterations are much slower after the CPU has dropped to a low
power state; without it the first trial was systematically faster than the
rest ([cli.md](cli.md#what-run-measures)). Trials are ordered by a scheduler
that, for several configurations, **rotates the order** every round, so that no
configuration always runs first or last
([architecture.md](architecture.md#trial-loop-and-scheduling)). `edgebench run`
benchmarks one configuration; benchmarking several configurations in one
interleaved run is planned.

Node placement (which execution provider runs each node) is recorded after the
last trial in a separate session with verbose logging, so it can never warm up
or slow down a timed session.

## Configuration that is always explicit

- **CPU thread count.** Every CPU run sets the intra-op thread count (default 4)
  and records it. ONNX Runtime's own default (one thread per core) made
  single-image latency bimodal under light background load on the reference
  machine ([feasibility, section 7.1](feasibility.md#71-npu-and-cpu-from-python)).
- **Run binding.** The default is IoBinding with input and output buffers
  allocated once (`config.run_binding = "iobinding_preallocated"`), so the timed
  call does not include per-call tensor allocation. Plain `Session::Run` is
  available with `--binding run`. Both give bit-identical outputs; in C++ on
  the NPU, IoBinding had the lower host overhead per inference (43 against
  52 µs, [feasibility, section 7.3](feasibility.md#73-host-overhead-breakdown)).
- **Graph optimization level, input kind and seed, warm-up and iteration
  counts** are recorded in `config`.

## Statistics

Per trial and pooled over all trials (`aggregate`):

- **Percentiles** use the nearest-rank method (p = rank ⌈p·n/100⌉ of the
  sorted values), so every reported percentile is an observed latency. p95 is
  reported only with at least 20 samples, p99 with at least 100.
- **Standard deviation** is the sample standard deviation (n − 1), reported with
  at least 2 samples.
- **Throughput** is 1000 / mean (sequential, one inference at a time).
- **Trial spread**: the range and coefficient of variation of the trial means
  (`aggregate.trial_variation`). A large spread means the run should not be
  summarized by one number.

The raw per-iteration latencies are written to a CSV file next to the result,
linked by SHA-256, so any other statistic can be recomputed.

## System state

CPU latency on the reference machine moved by about 2× between runs on the same
day with identical settings
([feasibility, section 7.1](feasibility.md#71-npu-and-cpu-from-python)). A
result is only interpretable together with the state the machine was in, so
every run records:

- power source, Windows power mode, power plan, battery charge and battery
  saver, at the start and at the end (a change produces a
  `power_state_changed` warning);
- system-wide CPU utilization during the idle before each trial (more than 10%
  produces a `background_cpu_load_high` warning);
- the operating system, CPU, device, ONNX Runtime version and the full path of
  every ONNX Runtime and QNN library that was loaded.

**Memory** is the process's own working set and private bytes (peak values,
and the working-set change across session creation). Memory used by drivers or
on an accelerator is not included.

## What EdgeBench does not claim

- **No power or energy measurements.** Power mode and source are recorded as
  context only.
- **No NPU utilization.** NPU execution is verified by node placement (every
  node on the requested execution provider), not by hardware counters.
- **No preprocessing or end-to-end application latency.**
- **No thermal or sustained-load behaviour.** Runs are short; long-running
  throughput under thermal limits has not been characterized.
- **No cross-device generalization.** Results describe one machine in one
  recorded state. They are comparable between configurations measured in the
  same run, and with care between runs whose recorded state matches.

See [feasibility, section 11](feasibility.md#11-limitations) for the
limitations of the evidence itself.
