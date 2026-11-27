# NPU runs

How EdgeBench runs models on the Snapdragon NPU (Qualcomm QNN execution
provider, HTP backend) and what to know when reading NPU numbers. Every option,
result field and exit code mentioned here is specified in
[cli.md](cli.md#npu-runs); the measurement rules are in
[methodology.md](methodology.md).

## Run doctor first

```powershell
.\build\arm64-release\bin\Release\edgebench.exe doctor
```

`doctor` checks, in order, everything an NPU run depends on, and runs every
check even after a failure: the ONNX Runtime and QNN files of the bundle against their pinned
hashes, the QNN EP's registration and the backend versions it reports, the
device list and the NPU a run would select, two tiny probe models run on the
NPU (INT8 and FP16) whose outputs must equal their expected bytes exactly, an
EP context built, loaded and compared, the cache directory, the power settings
and the background CPU load. Each check is a pass, a warning or a failure with
a code and a hint; the exit code is that of the first failure, so a failing
doctor predicts how `run` would fail. A warning on `power` (battery, or a power
mode other than Best performance) or `background_load` does not stop a run, but
a result measured that way is not comparable with one measured on AC power, in
Best performance, on an idle machine ([cli.md](cli.md#edgebench-doctor)).

## Device and bundle

`--provider npu` registers the QNN EP from the bundle (the files next to
`edgebench.exe`) and selects the one QNN device of type NPU; none or several is
exit 3. The QNN EP lists the NPU from hardware discovery alone, even when the HTP
backend files are missing, so an NPU in the device list proves nothing about
the backend. edgebench therefore checks that `QnnHtp.dll` and
`QnnHtpPrepare.dll` are in the bundle before the first session (exit 4), and
`QnnSystem.dll` too when a run uses EP contexts.

The architecture-specific stub (`QnnHtpV73Stub.dll` on this device) cannot be
checked in advance: which one the backend loads depends on the hardware. Without
it in the bundle, the backend does not fail. It loads the copies that the NPU
display driver keeps in the Windows driver store and runs with them, so the
run would silently use a backend edgebench did not pin. That is why the loaded
ONNX Runtime and QNN modules are checked after the EP registration, after every
session creation (the HTP DLLs load at that point) and while the placement and
profiling sessions exist: any such module outside the bundle directory stops the
run with `module_outside_bundle` (exit 4). Each check is recorded with its stage
in `execution.verification_details.module_checks`.

## Strict mode

By default an NPU session is strict: `session.disable_cpu_ep_fallback` is `1`,
so ONNX Runtime refuses to create a session in which any node would run on the
CPU. A refusal fails the run with `strict_mode_refusal` (exit 5) and lists the
nodes the QNN EP could not take (`error.unsupported_nodes`). A strict result
is therefore all-NPU by construction; the placement session after the trials
confirms it (every node on the QNN EP, `full_device_execution_verified`) and
records the QNN EP's own node validation (`execution.qnn_validation`).

`--no-strict` allows the fallback. The result then records `strict_mode` false
and carries a `strict_mode_off` warning, and a `cpu_fallback_nodes` warning
naming the nodes that ran on the CPU: such a result measures a CPU/NPU mix, not
the NPU. A backend that cannot be loaded is told apart from a refusal (ONNX
Runtime reports both the same way) by what the QNN EP logged during the session
creation: `qnn_backend_load_failed`, exit 4.

## Precision

The NPU computes INT8 or FP16, never FP32.

- An INT8 QDQ model (static quantization) runs as stored:
  `execution.effective_precision` is stored `int8`, compute `int8`, source
  `model`.
- An FP32 model needs `--npu-fp16`, which passes `enable_htp_fp16_precision`
  `1`: stored `fp32`, compute `fp16`, source `enable_htp_fp16_precision`, and a
  `model.precision_notes` entry. Without the flag the run is refused
  (`npu_fp32_needs_fp16`, exit 2) rather than silently computed in FP16.
- `--npu-fp16` with an INT8 model, and any other stored precision, are usage
  errors (exit 2).

On the CPU the compute precision is the stored one.

## Provider options

edgebench passes the QNN EP exactly these provider options and records them in
`config.ep_options`; every other option stays at the EP's default, which the
result says with `runtime.qnn.unlisted_ep_options` `ep_default`:

| Option | Value | Set by |
|---|---|---|
| `backend_path` | absolute path of the bundled `QnnHtp.dll` | always |
| `htp_performance_mode` | `burst` | `--htp-performance-mode` |
| `enable_htp_fp16_precision` | `1` | `--npu-fp16` only |
| `htp_graph_finalization_optimization_mode` | `0`, `1`, `2` or `3` | `--htp-graph-finalization-mode` only |

**Graph finalization.** `--htp-graph-finalization-mode` sets how much the HTP
backend optimizes the graph when it finalizes it, with the values the QNN EP
documents: `0` (the EP default), `1` (faster preparation, a less optimized
graph), `2` (longer preparation, a more optimized graph), `3` (longest
preparation). Without the option nothing is passed and the backend's default
applies, which is what every result without it measures. The option is recorded
in `config.ep_options`, is part of the EP context cache key like every option,
and decides which batch-equivalence facts apply to the run (below). Mode 1 is
what makes the INT8 image models compute the same outputs at batch 8 as at
batch 1, at a latency cost ([Batch size and INT8 outputs](#options)); it is an
option for experiments, not the default.

## Initialization

Creating an NPU session from the ONNX model compiles it for the HTP. A session
created from an EP context (the precompiled graph the QNN EP can write) skips
the compilation. `--init-breakdown` measures the three kinds of session
creation in every trial, each after the same idle period as the trial, and
before the trial's timed session (`timing.trials[].session_init`, summarized in
`timing.session_init.summary`):

- **cold** (`cold_ms`): a session from the model, compiled from scratch;
- **build** (`context_build_ms`): a session that compiles the model and writes
  an EP context into a scratch directory of that trial;
- **cached** (`cached_init_ms`): a session from that scratch context.

Cold and build times are close to each other: writing the context costs little
compared with compiling. Much of a cold session creation is not the graph
compilation at all: the QNN profile (below) shows the graph finalization as a
small part of it, and the rest, a cost of roughly the same size at batch 1 for
every model from the tiny autoencoder to ResNet-18, is the backend and device
setup; larger batches compile longer, FP16 more so than INT8. Cached sessions
avoid most of the cost, and their creation time hardly depends on the model or
the batch size. Initialization times vary between trials and
between sessions of the machine, so they are reported with their spread.

## EP context cache

`--ep-cache` decides where the timed sessions come from:

- `off` (default): every timed session is compiled from the model
  (`ep_context_state` `cold`);
- `build`: delete this key's cache entry, build it once before the trials,
  create every timed session from it;
- `use`: create the timed sessions from the entry whose key matches; build it
  first if there is none (`hit` false), load it if there is (`hit` true).

**Location.** `%LOCALAPPDATA%\EdgeBench\qnn-context` by default, or
`--ep-cache-dir`; one subdirectory per key, holding the context model, its
context binary and a manifest with every file's size and SHA256. edgebench only
deletes files listed in a manifest it wrote. `doctor` reports the directory and
its stale entries, and `doctor --prune-cache` deletes them.

**Key.** The key covers everything that could make a stored context wrong for
the run: the model file's SHA256, the ONNX Runtime, QNN EP and QNN SDK
versions, the HTP backend file version, every EP option passed (graph
finalization mode included), the session settings that affect compilation, the
graph optimization level and the device. A change to any of them gives a new
key, so a stale context is never loaded.

**Stale entries.** An entry is stale when it was built by another ONNX
Runtime, QNN EP, QNN SDK or HTP backend version, or for another device: no run
on this machine can load it again. A run that builds or uses the cache deletes
the stale entries of its own model and configuration; `doctor --prune-cache`
deletes every stale entry. Entries for different options of the same model
file (for example `--htp-graph-finalization-mode 1` and the default) are not
stale: they coexist, and each stays a cache hit.

**Verification.** Before the trials of a `build` or `use` run, the engine runs
its input through a session compiled from the model and through a session from
the context, and compares the output bytes: `verified_equal_outputs` true, or
the run fails (`ep_context_outputs_differ`, exit 1). A cache file that is
missing or changed, or a context ONNX Runtime cannot load, fails the run (exit
3) with a hint to rebuild it with `--ep-cache build`; nothing is rebuilt
silently.

**The "suboptimal" note.** Every session from a context makes the QNN EP log
that the context is supported but suboptimal and should be recompiled; the
result gets an `ep_context_recompilation_suggested` note. The cause is known:
the EP rates a context optimal only if the VTCM size stored in it equals the
device's, and without the `vtcm_mb` option it stores 0. Compiling with the
device's VTCM size makes the verdict optimal with bit-identical outputs and the
same initialization and latency, but nothing reports the device's VTCM size
before a context exists, and a value larger than the device's makes the context
unusable, so edgebench does not set it. The note does not affect the results:
the context's outputs are verified equal to a compiled session's. The
compatibility info and verdict are in `config.ep_context_cache.compatibility`
([cli.md](cli.md#ep-context-cache-and-initialization)).

## Accelerator time and latency

The timed latency of an NPU inference is what an application waits for: ONNX
Runtime and the QNN EP on the CPU, the transfer to the NPU and back, and the
NPU's own work. `--profile-accelerator` adds a separate session with QNN
profiling after the trials and records the backend's accelerator time per
inference (`timing.profiled_pass`): the part the HTP itself spends. It comes
from another session than the timed ones and never enters their statistics
([methodology.md](methodology.md#accelerator-time)).

The difference between the median latency and the median accelerator time is
the host and transfer overhead of an inference. It is roughly constant per
inference, so it matters most for small models: for a tiny model such as the
autoencoder it is a large share of the latency, so a CPU that needs no
transfer can be faster than the NPU on such a model, and the NPU's latency says
as much about the host path as about the NPU. Comparing an NPU
latency with a CPU latency compares what an application gets, overhead
included; the accelerator time says how much of it the NPU is responsible for.
An accelerator time above the latency means one of the two is wrong and gets an
`accelerator_time_exceeds_latency` warning.

## Batch equivalence

A result at batch 4 or 8 records whether the file computes, per item, exactly
what its batch-1 file computes on the same provider and compute precision, and
if not, whether any prediction changes (`execution.batch_equivalence`, from the
committed `models/batch_equivalence.json`; [models.md](models.md#batch-equivalence)).
A run whose predictions differ from batch 1 gets a `batch_not_equivalent`
warning and must not be averaged or compared with batch-1 results as the same
computation.

On the NPU this matters for the INT8 image models at batch 8 with the default
graph finalization: their outputs differ from batch 1 on every image
([below](#batch-size-and-int8-outputs)), and the published eval-full accuracy
at batch 8 is compared with batch 4 in [accuracy.md](accuracy.md#batch-8-on-the-npu).
The facts are measured per set of compile-relevant EP options (every option
except `backend_path`, `htp_performance_mode` and `enable_htp_fp16_precision`):
with `--htp-graph-finalization-mode 1` the same files have their own entries
(batch 8 equal to batch 1), and with any option combination that has no entries
the result records batch equivalence as not available, with a
`batch_equivalence_options_not_covered` note, rather than reusing the default
options' answer.

## Batch size and INT8 outputs

### Observation

On the NPU, the INT8 image models (MobileNetV2, ResNet-18, EfficientNet-Lite0)
give logits at batch 4 that are bit-identical to batch 1, but at batch 8 every
image's logits differ from batch 1, and the top-1 class changes on a few of the
1,000 eval-quick images ([models.md](models.md#batch-equivalence),
[accuracy.md](accuracy.md)). The same files on the CPU, the FP32 files on the NPU
(FP16 compute) and the INT8 autoencoder on the NPU are batch-consistent, so the
quantized graph itself computes the same thing at every batch size; the
difference comes from how the HTP executes it. The C++ engine and Python ONNX
Runtime give the same b8 outputs bit for bit.

`scripts/investigate_batch_equivalence.py` regenerates everything below and
writes it to [`docs/findings/npu_int8_batch_size.json`](findings/npu_int8_batch_size.json)
(about 20 minutes on the NPU; nothing under `models/` changes). All sessions are
strict QNN HTP sessions with the reference options (burst performance mode).
Latencies are a quick check (interleaved, 10 rounds of 50 runs, medians), not a
benchmark.

### Where the threshold is

The INT8 files were built at batch sizes 2, 3, 5, 6, 7 and 16 with the shipped
recipe, reusing the batch-1 calibration ranges: every batch size has exactly the
shipped scales and zero points, and the rebuilt b1 and b4 graphs are
byte-identical to the shipped files. Each batch size ran the 8 reference images
(repeated to fill a batch); the table counts the images whose logits equal the
batch-1 reference bit for bit, next to the largest uint8 activation in the graph
(the first expanded block's output for MobileNetV2 and EfficientNet-Lite0, the
first convolution's output for ResNet-18). The device's VTCM is 8 MiB.

| Batch | MobileNetV2 | ResNet-18 | EfficientNet-Lite0 | Largest activation, MobileNetV2 | Largest activation, ResNet-18 | Largest activation, EfficientNet-Lite0 |
|---|---|---|---|---|---|---|
| 1 | 8 of 8 | 8 of 8 | 8 of 8 | 1.15 MiB | 0.77 MiB | 1.15 MiB |
| 2 | 8 of 8 | 8 of 8 | 8 of 8 | 2.30 MiB | 1.53 MiB | 2.30 MiB |
| 3 | 8 of 8 | 8 of 8 | 8 of 8 | 3.45 MiB | 2.30 MiB | 3.45 MiB |
| 4 | 8 of 8 | 8 of 8 | 8 of 8 | 4.59 MiB | 3.06 MiB | 4.59 MiB |
| 5 | 8 of 8 | 8 of 8 | 8 of 8 | 5.74 MiB | 3.83 MiB | 5.74 MiB |
| 6 | 8 of 8 | 8 of 8 | 8 of 8 | 6.89 MiB | 4.59 MiB | 6.89 MiB |
| 7 | 0 of 8 | 0 of 8 | 0 of 8 | 8.04 MiB | 5.36 MiB | 8.04 MiB |
| 8 | 0 of 8 | 0 of 8 | 0 of 8 | 9.19 MiB | 6.12 MiB | 9.19 MiB |
| 16 | 0 of 8 | 0 of 8 | 0 of 8 | 18.38 MiB | 12.25 MiB | 18.38 MiB |

The smallest batch size that differs is 7 for all three models. For MobileNetV2
and EfficientNet-Lite0 the largest activation crosses 8 MiB between b6 and b7,
but for ResNet-18 it is only 5.36 MiB at b7, so the threshold does not follow
the activation size. From b7 on the outputs are the same at every batch size
(b7, b8 and b16 give identical logits per image) and do not depend on an
image's position in the batch; repeated runs are identical.

### Where it starts

Of the 33 activations of ResNet-18 (the output of every activation
DequantizeLinear), compared at b1 and b7, only the last one, `logits`, differs:
every intermediate tensor, including the pooled features that enter the
classifier, is bit-identical. Each comparison is a truncated model ending at
that tensor. Truncation can change how the HTP compiles the end of a graph, so
the classifier tail was also cut out as sub-models fed with uint8 activations:
the real activations entering GlobalAveragePool for the 8 reference images,
and, for the Gemm alone, 64 random uint8 vectors. The table counts the outputs
that differ from the same sub-model at b1:

| Model | Sub-model | b2 | b4 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|
| MobileNetV2 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| MobileNetV2 | GlobalAveragePool + Flatten | 0 | 0 | 129 of 10,240 | 129 of 10,240 | 965 of 10,240 |
| MobileNetV2 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,671 of 8,000 | 1,671 of 8,000 |
| MobileNetV2 | Gemm only | 0 | 0 | 0 | 12,180 of 64,000 | 12,180 of 64,000 |
| ResNet-18 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| ResNet-18 | GlobalAveragePool + Flatten | 0 | 0 | 0 | 0 | 0 |
| ResNet-18 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,713 of 8,000 | 1,713 of 8,000 |
| ResNet-18 | Gemm only | 0 | 0 | 0 | 13,683 of 64,000 | 13,683 of 64,000 |
| EfficientNet-Lite0 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| EfficientNet-Lite0 | GlobalAveragePool + Flatten | 0 | 0 | 142 of 10,240 | 142 of 10,240 | 1,153 of 10,240 |
| EfficientNet-Lite0 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,866 of 8,000 | 1,866 of 8,000 |
| EfficientNet-Lite0 | Gemm only | 0 | 0 | 0 | 6,615 of 64,000 | 6,615 of 64,000 |

The final fully connected layer (ONNX Gemm, QNN FullyConnected) reproduces the
threshold on its own: its sub-model is bit-identical up to b6 and differs from
b7, and the tail from GlobalAveragePool to the logits differs exactly as much as
the full model does (1,671 / 1,713 / 1,866 of 8,000, see below). The
"GlobalAveragePool + Flatten" sub-models of MobileNetV2 and EfficientNet-Lite0
also differ, already at b6, but only when the Flatten output is the end of the
graph: in the full model and in the tail sub-model the b6 logits are
bit-identical, so this is an effect of cutting the graph there, not part of the
divergence. The QNN graphs the execution provider builds for the Gemm
sub-model at b6 and b7 (`dump_json_qnn_graph`) differ only in the batch
dimension of 3 tensors: the choice that changes the results is made inside the
HTP backend when it finalizes the graph.

How much to trust this: the full-model layer scan covers ResNet-18 only; for
the other two models only the tail was isolated. Their pooled features (the
GlobalAveragePool output) are bit-identical at b1 and b7, and the tail
sub-model reproduces the full model's logits difference exactly, so an earlier
divergence that disappears before the classifier cannot be excluded for them,
but nothing points to one.

### How large the differences are

In units of the logits' quantization scale (LSB), every differing element is +1
LSB: at b7 and above the HTP never produces a smaller value than at b1, and
never one more than 1 LSB away. Full models, b8 against b1, 8 reference images:

| Model | Output scale | Outputs | +1 LSB | Top-1 changes |
|---|---|---|---|---|
| MobileNetV2 | 0.0570 | 8,000 | 1,671 (20.9) | 0 |
| ResNet-18 | 0.1686 | 8,000 | 1,713 (21.4) | 0 |
| EfficientNet-Lite0 | 0.0823 | 8,000 | 1,866 (23.3) | 0 |

(+1 LSB: count and percentage of the outputs.) A one-sided shift is not what a
different accumulation order gives (that moves values both ways); it is what a
different requantization of the Gemm's int32 accumulator gives. To see which
batch size is closer to the QDQ definition, the Gemm was computed in exact
integer arithmetic for the 64 random vectors (int32 accumulator with the int32
bias, scaled by input scale x per-channel weight scale / output scale, plus the
output zero point; the bias scale is input scale x weight scale, as the quantizer
writes it) and the NPU outputs compared with that exact value, in output LSB,
over the outputs not clipped to the uint8 range:

| Model | Batch | Mean error (LSB) | Standard deviation (LSB) | Range (LSB) | Exactly rounded |
|---|---|---|---|---|---|
| MobileNetV2 | 1 | -0.001 | 0.29 | -0.56 to 0.56 | 97.8 |
| MobileNetV2 | 8 | 0.238 | 0.51 | -0.56 to 1.55 | 74.7 |
| ResNet-18 | 1 | -0.001 | 0.29 | -0.55 to 0.55 | 97.7 |
| ResNet-18 | 8 | 0.239 | 0.51 | -0.55 to 1.55 | 74.5 |
| EfficientNet-Lite0 | 1 | 0.005 | 0.29 | -0.56 to 0.56 | 97.7 |
| EfficientNet-Lite0 | 8 | 0.260 | 0.52 | -0.56 to 1.55 | 73.0 |

("Exactly rounded": percentage of outputs equal to the exact value rounded to
the nearest integer.) At b1 the HTP rounds to nearest: the error is centred on
zero with the spread of plain rounding (0.29 = 1/sqrt(12)), and the outputs that
are not exactly rounded lie within 0.06 LSB of a rounding boundary. From b7 on,
the Gemm output is on average about a quarter LSB higher, up to 1.55 LSB above
the exact value (b1: at most 0.56). The b7+ kernel requantizes less precisely;
nothing indicates an error (no output is off by more than 1 LSB from b1, and
the shift is the same for every model and input). On the
8 reference images no top-1 class changes; on the 1,000 eval-quick images 6, 10
and 5 do (below and [models.md](models.md#batch-equivalence)).

### Options

Full models at b1 (control), b6, b7 and b8, each option added to the reference
options ("same": all 8 images of all three models equal the batch-1 reference
bit for bit; "differs": none does):

| Option | b1 | b6 | b7 | b8 |
|---|---|---|---|---|
| `default` | same | same | differs | differs |
| `vtcm_mb=2` | same | same | differs | differs |
| `vtcm_mb=4` | same | same | differs | differs |
| `vtcm_mb=8` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=0` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=1` | same | same | same | same |
| `htp_graph_finalization_optimization_mode=2` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=3` | same | same | differs | differs |
| `htp_performance_mode=balanced` | same | same | differs | differs |
| `htp_performance_mode=power_saver` | same | same | differs | differs |
| `htp_performance_mode=high_performance` | same | same | differs | differs |

A smaller VTCM does not move the threshold, and the performance mode changes
no output (as expected). Only graph finalization mode 1 (the quickest
finalization, fewer graph optimizations) makes b7 and b8 bit-identical to b1;
it does the same for the Gemm sub-model on its own. On all 1,000 eval-quick
images, and its latency cost (median ms per inference of the whole batch;
mode 1 at b1 is also bit-identical to the default on all 1,000 images):

| Model | Default b8: top-1 changes | Mode 1 b8: bitwise equal to b1 | b1 default | b1 mode 1 | b8 default | b8 mode 1 | Mode 1 cost at b8 |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | 6 of 1,000 | 1,000 of 1,000 | 0.32 | 0.32 | 1.98 | 2.29 | +15% |
| ResNet-18 | 10 of 1,000 | 1,000 of 1,000 | 0.37 | 0.44 | 2.18 | 2.83 | +29% |
| EfficientNet-Lite0 | 5 of 1,000 | 1,000 of 1,000 | 0.34 | 0.35 | 2.22 | 2.45 | +10% |

Mode 1 is not the default: edgebench keeps the backend's default finalization,
and NPU INT8 runs at b8 carry the `batch_not_equivalent` warning. Mode 1 is
available as `--htp-graph-finalization-mode 1`
([Provider options](#provider-options)); its runs are matched with their own
batch-equivalence entries, which confirm on the references and on all 1,000
eval-quick images that b4 and b8 compute exactly the b1 outputs
([models.md](models.md#batch-equivalence)).

### Conclusion

The NPU INT8 image models differ at batch 7 and above because of one layer, the
final fully connected layer: from 7 rows on, the HTP backend's default graph
finalization compiles QNN FullyConnected into a kernel whose requantization of
the output rounds up to 1 LSB higher than at 1 to 6 rows (+0.24 LSB on
average). The quantized graph and the QNN graph are the same at every batch
size; nothing before the classifier changes (checked layer by layer for
ResNet-18). This is a different computation, not an error. It is not a memory
effect: the threshold is the same for all three models although their
activations differ in size, and the VTCM size does not move it. The
autoencoder is not affected because it has no fully connected layer, and FP16
runs use different kernels.

Confidence: high that the final Gemm is where b7+ differs and that
finalization mode 1 removes it (reproduced in the full models, in the isolated
Gemm, and on 1,000 images). What the backend's kernel does internally (which
kernel, why from 7 rows) is inferred from the outputs; the QNN SDK does not
report it, and settling it would need the backend's per-op profiling or
Qualcomm's documentation of the FullyConnected kernels.
