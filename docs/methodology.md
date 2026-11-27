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

Where normalization happens is part of the model, so every result records it
as `model.normalization`: `external` (done before the model, not timed),
`in_graph` (Sub/Div nodes inside the model, timed) or `folded` (merged into the
first convolution, so the model takes [0, 1] input and the timed run includes
whatever the fold adds). Only results with the same normalization placement
measure the same work ([models.md](models.md#normalization-folded-models)).

Latency is only half of a CPU vs NPU comparison: the two devices compute an
INT8 model with different kernels, so their outputs differ. The accuracy of
every variant on each device, and whether the NPU's accuracy on the same INT8
file matches the CPU's, is measured separately and needs no idle machine
([accuracy.md](accuracy.md)).

Session creation is measured separately (`timing.session_init`) and is never
mixed into the iteration latencies.

## Initialization and the EP context cache

On the NPU, creating a session means compiling the graph for the HTP (about
2 s for the models here) unless an EP context, the compiled graph the QNN EP
can write to disk, already exists; a session from a context takes about 0.3 s.
These are different costs and are reported separately, each from repeated
creations (`--init-breakdown`, one of each per trial, after the same idle
period as the trial):

- **cold** (`cold_ms`): a session created from the ONNX model with no context;
  what every session costs without a cache;
- **context build** (`context_build_ms`): a session that compiles the model and
  writes a context; a fresh build in every trial, not a rewrite of a shared one;
- **cached** (`cached_init_ms`): a session created from that context.

The steady-state latency comes from the trial's timed session in every case,
created as `--ep-cache` says: compiled from the model (`off`, the default) or
from the cache (`build`, `use`). A context depends on more than the model, so
the cache is keyed by everything it depends on: the model file's SHA256, the
ONNX Runtime and QNN EP versions, the QNN SDK version the backend reports at run
time, the HTP backend file version, every EP option, every session config
entry that changes compilation (strict mode, embed mode), the graph
optimization level, and the device's vendor and device id. A change to any of
them gives a new key, so a stale context is never loaded; entries of the same
model and configuration built by another runtime, backend or device are
deleted, and entries for other options coexist. Before a run uses a context, one session
from the context and one compiled from the model run the same input, and their
outputs must be bit-identical (`verified_equal_outputs`); a corrupt or
unloadable context fails the run rather than being rebuilt behind the user's
back. The rules and file layout are in
[cli.md](cli.md#ep-context-cache-and-initialization).

## Accelerator time

The timed latency is what an application waits for: the host-side work of
ONNX Runtime and the QNN EP, the RPC to the NPU and the NPU's own work. With
`--profile-accelerator` a result also records the **accelerator time**: the
QNN profile's "Accelerator (execute) time" per inference, the part of an
inference the HTP itself spends, from a separate session with QNN profiling
(`basic`) created after the timed trials (`timing.profiled_pass`).

It is not a latency and is never compared with one as if it were:

- it comes from another session than the timed ones (the same model or EP
  context, the same options plus profiling), so nothing measured in it is part
  of the trials, their statistics or the initialization times;
- profiling adds host-side work to every inference of that session (QNN
  basic profiling added about 12 ms of host time per inference in the
  feasibility work), so the profiled session's own wall time is not usable;
  the accelerator time is reported by the backend and is not affected;
- the difference between the median timed latency and the median accelerator
  time is the host and transfer share of an inference. If the accelerator time
  is the larger one, something is wrong with one of the two numbers, and the
  result gets an `accelerator_time_exceeds_latency` warning.

The profile also lists the profiled session's graph finalization (or, from an
EP context, the loading of the context binary). Those are a small part of
session creation: about 0.34 s of a 2.2 s cold MobileNetV2 INT8 session, and
30 to 60 ms of a 0.35 to 0.45 s cached one. Most of the roughly 2 s fixed cold
cost (backend and device setup) does not appear in the profile.

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
  summarized by one number: a coefficient of variation above 10%, or a slowest
  trial mean more than 1.25× the fastest, produces a `trial_spread_high`
  warning.

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
- background CPU load before the first trial, between trials (the idle before
  each trial) and during each trial's measured iterations; more than 10% in any
  sample produces a `background_cpu_load_high` warning that says when;
- the operating system, CPU, device, ONNX Runtime version and the full path of
  every ONNX Runtime and QNN library that was loaded (a library from outside
  the bundle directory stops the run).

During a trial, the benchmark's own threads keep the CPU busy, so the
system-wide utilization (`GetSystemTimes`) alone would mostly measure the
benchmark itself. EdgeBench therefore subtracts its own process's CPU time over
the same window (`GetProcessTimes`, all threads) from the system's busy time
and divides the remainder by the system's total time, which is the window times
the number of logical processors. What is left is the load from other
processes. The idle samples are computed the same way, so a session that is
still being torn down never counts as background load. The result's
`system.load` records the pre-run sample; the during-trial values are reported
in the summary and in warnings. Every sample above 10% is recorded with the
top other processes by share, to find the source of the load.

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
