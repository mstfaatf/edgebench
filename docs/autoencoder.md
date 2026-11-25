# Time-series anomaly-detection autoencoder

The trained model is part of the model pipeline with the same guarantees as
the image models: a pinned source, fixed shapes, recipe metadata and sidecars,
manifest lineage, INT8 with the default recipe, a strict NPU check, reference
outputs and a bit-exact match of the C++ engine. Its anomaly-detection accuracy
on CPU and NPU is published in
[`benchmarks/results/published/accuracy/anomaly/`](../benchmarks/results/published/accuracy/anomaly/).
Latency is not published yet.

The image models answer "which class is this?". The autoencoder adds a
different kind of workload: a small convolutional model that reconstructs a
window of a sensor series. A window it cannot reconstruct well is flagged as
anomalous. The model is tiny (about 9,500 parameters, ~40 KB as FP32 ONNX), so
on the NPU fixed per-inference costs dominate the measured latency rather than
arithmetic. That makes it a useful counterpoint to the image models when
comparing CPU and NPU.

## Dataset

[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB),
`data/realKnownCause/machine_temperature_system_failure.csv`: temperature of an
internal component of a large industrial machine, one reading every 5 minutes,
2013-12-02 to 2014-02-19 (22,695 points). It has four labelled anomaly windows
from `labels/combined_windows.json`, including a planned shutdown and the
catastrophic failure at the end. NAB's description says the anomaly causes are
known, not hand-labelled.

| | |
|---|---|
| NAB commit | `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` |
| Series SHA256 | `92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4` (732,223 bytes) |
| Labels SHA256 | `1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1` (15,359 bytes) |
| License | MIT (the repository license, which covers its `data/` and `labels/`; NAB was AGPL-3.0 before this commit) |

The files are downloaded, never committed or redistributed:

```powershell
.\.venv\Scripts\python.exe scripts\get_data.py --dataset nab
```

This downloads both files from the pinned commit into `data/nab/`, checks their
SHA256 on every run, rebuilds the windows and splits, and compares them with the
committed manifest [`datasets/nab-machine-temperature/splits.json`](../datasets/nab-machine-temperature/splits.json).

**One data quirk.** At 2014-01-07 02:55 the timestamps step back to 02:00, so
one hour of timestamps appears twice (a clock correction). The series is used
in file order, as NAB itself does. The repeated hour is far from every label
window, and label mapping refuses any window that would cover non-contiguous
points.

## Windows and splits

All of this is computed by [`scripts/common/timeseries.py`](../scripts/common/timeseries.py),
which uses only the Python standard library to decide which values a model
sees (NumPy only builds the arrays). That makes the manifest's hashes identical
on every platform, and it means the training notebook runs this same code
rather than a copy.

- **Window length 128** (10 h 40 min). This is long enough to contain the
  slow, hours-long temperature swings of normal operation and short against
  the 567-point label windows, so windows lie entirely inside an anomaly. It
  is also a power of two, so two stride-2 convolutions halve it cleanly
  (128 → 64 → 32) and two stride-2 transposed convolutions restore it.
- **Stride 1** in every split, which gives a reconstruction score for every
  time step and the largest number of training windows (8,281). The model is
  small enough that this costs little.
- **A window is anomalous** if any of its points lies inside a labelled window.
- **Time-ordered splits.** Each split is a set of contiguous index ranges and
  windows never cross a range boundary. Only 2,126 points come before the
  first anomaly, too few for both training and validation. Training and
  validation therefore come from the longest anomaly-free stretch, between
  the second and third anomaly. Training comes first, then validation; the
  last 2 days of the stretch (576 points) go to the test split, so the test
  data has normal windows right before the third anomaly. The test split is
  everything else, before and after the stretch, and holds all four anomalies.

| Split | Index ranges | Points | Windows | Windows touching a label |
|---|---|---|---|---|
| train | [4270, 12678) | 8,408 | 8,281 | 0 |
| validation | [12678, 15481) | 2,803 | 2,676 | 0 |
| test | [0, 4270) and [15481, 22695) | 11,484 | 11,230 | 2,649 |

- **Normalization:** `(x - mean) / std` with the mean and population standard
  deviation of the 8,408 training points (`math.fsum`, correctly rounded):
  mean 89.97448888370599, std 8.061133940337553. This is external
  normalization, the counterpart of the image models' mean/std, and the
  model's input is the normalized float32 series.
- **Validation** is used for early stopping and later for choosing the
  detection threshold, never for training.
- **INT8 calibration:** 200 windows evenly spaced over the training windows.
- **Reference windows:** 8 fixed test windows, one centred in each anomaly
  and 4 evenly spaced normal ones.

The manifest records the source files and their SHA256, the anomaly windows
as index ranges, the split ranges and window counts, the normalization
constants, the calibration and reference window starts, and the SHA256 of the
raw float64 values and of the normalized float32 series that every window is
cut from. `scripts/tests/test_timeseries.py` tests the logic on synthetic
series in CI. Tests on the real files skip visibly when the data is not
downloaded.

## Why training happens in Colab

TensorFlow has no usable native build for Windows on Arm, and this project
does not use x64 emulation. So the Keras model is trained in Google Colab with
[`notebooks/train_autoencoder.ipynb`](../notebooks/train_autoencoder.ipynb),
which:

- installs pinned versions first: tensorflow 2.21.0, keras 3.15.1, tf2onnx
  1.17.0, onnx 1.23.2, onnxruntime 1.32.0 (the version used on the target
  machine);
- clones this repository at a pinned commit, runs `get_data.py --dataset nab`,
  and loads the windows with `timeseries.load_split_windows()`, which raises if
  anything differs from the committed manifest;
- builds the model from the layer list in
  [`scripts/common/autoencoder.py`](../scripts/common/autoencoder.py), the same
  list the NPU check below used;
- trains with fixed seeds, TensorFlow op determinism and one CPU thread, using
  MSE, Adam (learning rate 1e-3), batch 64, and early stopping on the
  validation reconstruction error (patience 10, best weights restored);
- exports with `tf2onnx.convert.from_function` at fixed batch sizes 1, 4 and 8,
  opset 17, input `input` and output `reconstruction`. It checks every file with
  the ONNX checker and the static-shape check of the image pipeline, and
  compares onnxruntime with Keras on the validation and reference windows;
- writes a bundle: the three FP32 ONNX files, `training_record.json` (versions,
  seeds, curve, losses, architecture, data and split hashes, Keras vs ONNX
  differences), reference inputs and outputs (`.npy`), and `SHA256SUMS`.

Training is repeatable on the same machine type and library builds. Across
machines it is reproducible only to within floating-point noise, not bit for
bit. The exported bundle is therefore the source of truth, and it is committed
(see [the source bundle](#the-source-bundle)).

The notebook calls `tf2onnx.convert.from_function` on a `tf.function` with a
fixed input signature rather than `from_keras`. tf2onnx 1.17 is tested against
TensorFlow 2.13 to 2.15, which have no Python 3.12 wheels for Colab, and
converting a concrete function does not depend on Keras 2 model internals.

## Architecture and the NPU check behind it

Two candidate designs, both well under 1 MiB:

| Design | Layers | Parameters | FP32 ONNX |
|---|---|---|---|
| `conv1d` | Conv1D 32 (k7, s2, ReLU) → Conv1D 16 (k7, s2, ReLU) → Conv1DTranspose 16 (k7, s2, ReLU) → Conv1DTranspose 32 (k7, s2, ReLU) → Conv1D 1 (k7, linear); input `[batch, 128, 1]` | 9,505 | ~40 KB |
| `dense` | Dense 64 → 16 → 64 (ReLU) → 128 (linear); input `[batch, 128]` | 18,704 | ~76 KB |

QNN op support can be checked before training. `scripts/check_autoencoder_npu.py`
builds untrained ONNX stand-ins with random weights in the layouts tf2onnx
produces. TensorFlow runs Conv1D as a 2-D convolution on `[batch, 1, steps,
channels]`, so tf2onnx emits `Conv` / `ConvTranspose` with `1 x 7` kernels
between Transposes, and its optimizer usually cancels the Transposes between
layers. The check covers both outcomes and a plain 1-D `Conv` too. It quantizes
each stand-in with the default INT8 recipe (calibration on the 200 training
windows, b4/b8 reusing the b1 ranges), then creates a strict QNN HTP session
(CPU fallback disabled) and runs 32 validation windows on NPU and CPU. A test
checks the stand-ins against a NumPy implementation of the Keras layers
(padding and kernel layouts included).

| Stand-in | Layout | INT8 nodes | Strict session, b1/b4/b8 | QNN-validated nodes | Rejected | NPU b4/b8 == b1 |
|---|---|---|---|---|---|---|
| `conv1d` | Transposes at the ends only (optimized tf2onnx) | 35 | all created, all nodes on QNN | 33 | 0 | yes (bit-identical) |
| `conv1d_per_layer` | Unsqueeze/Transpose/Conv/Transpose/Squeeze per layer (unoptimized) | 83 | all created, all nodes on QNN | 57 | 0 | **no** (max diff 0.31) |
| `conv1d_native` | 1-D Conv on `[batch, channels, steps]` | 31 | all created, all nodes on QNN | 27 | 0 | yes |
| `dense` | MatMul + Add | 34 | all created, all nodes on QNN | 34 | 0 | yes |
| `dense_gemm` | Gemm | 22 | all created, all nodes on QNN | 22 | 0 | yes |

ONNX Runtime 1.32.0, onnxruntime-qnn 2.8.0, QNN backend 2.52.0; AC power,
Balanced power mode. The weights are random, so only op support and agreement
between runs are meaningful. Max |NPU − CPU INT8| on the 32 windows was 0.07
for `conv1d` and `conv1d_native`, 0.36 for `conv1d_per_layer` (both Dense
layouts: 0.36; outputs are of order 1).

**Choice: `conv1d`.** Every op in all three Conv1D layouts is accepted in strict
mode at batch 1, 4 and 8, with no rejection. The per-layer layout is the one
exception to "the NPU gives the same outputs at every batch size": like the
image models' INT8 graphs at batch 8, its b4/b8 outputs differ from b1.
tf2onnx produced yet another layout (below); the pipeline cleans it up and
checks the real files on the NPU again ([NPU](#npu)).

To repeat the check (writes to `benchmarks/results/local/autoencoder/`):

```powershell
.\.venv\Scripts\python.exe scripts\check_autoencoder_npu.py
```

## The source bundle

The Colab run's bundle is committed, about 130 KB, in
[`models/sources/conv1d_autoencoder/`](../models/sources/conv1d_autoencoder/):
the three FP32 ONNX files, `training_record.json`, the reference arrays
(`reference_inputs.npy`, `reference_keras.npy`, `reference_onnx_b1.npy`) and
`SHA256SUMS`. `.gitattributes` marks the directory binary, so no line-ending
conversion can change a hash. `IMPORTED_MODELS` in
[`scripts/common/model_registry.py`](../scripts/common/model_registry.py) pins
it: the SHA256 of each ONNX file, of the training record and of `SHA256SUMS`,
the notebook (`notebooks/train_autoencoder.ipynb` at commit `8d8f5a3`) and the
repository commit it cloned for data and code (`1e9eb1b`). The pipeline refuses
to run if any file differs from `SHA256SUMS`, from the pins or from the training
record's own export hashes.

| | |
|---|---|
| Training | Colab, Python 3.13.18, TensorFlow 2.21.0, Keras 3.15.1, tf2onnx 1.17.0, onnx 1.23.2, NumPy 2.4.6; seed 20261124 |
| Epochs | 143 run, best epoch 133 restored (366 s) |
| Reconstruction MSE | train 1.81e-5, validation 3.07e-5; validation window MSE p50 2.8e-5, p99 8.7e-5, max 3.5e-4 |
| FP32 files | b1 `a73173e6…f37c` (46,144 bytes), b4 `4aab5841…7379`, b8 `04b3cfd0…55f8a1` (46,189 bytes) |
| Keras vs ONNX (Colab, x64 ONNX Runtime) | max abs diff 2.4e-6 on all validation windows, 4.3e-6 on the reference windows; b4 and b8 equal b1 |

**What tf2onnx produced.** 26 nodes: `Conv` ×3, `ConvTranspose` ×2, `Add` ×5,
`Relu` ×4, `Reshape` ×2, `Squeeze` ×5, `Unsqueeze` ×5, and no `Transpose`.
Every layer is wrapped in its own `Unsqueeze` / `Squeeze` (axis 2) around a
`1 x 7` convolution on `[batch, channels, 1, steps]`, and each layer's bias is
a separate `Add` after the `Squeeze`. This layout was not among the stand-ins
checked before training.

## Graph clean-up

[`scripts/common/onnx_cleanup.py`](../scripts/common/onnx_cleanup.py) rewrites
exactly that pattern and nothing else:

1. each bias `Add` is folded into its convolution's bias input (the same values,
   reshaped from `[1, C, 1]` to `[C]`);
2. each `Squeeze` → `Relu` → `Unsqueeze` between two layers is removed; the
   `Relu` runs on the 4-D tensor, which is the same elementwise operation;
3. the `Unsqueeze` + `Reshape` at the input and the `Squeeze` + `Reshape` at the
   output each become one `Reshape` (a reshape of a reshape depends only on the
   final shape). The two reshapes get fixed names, because tf2onnx numbers its
   helper nodes differently for every batch size and the INT8 calibration
   ranges are shared across batch sizes by tensor name.

Anything that is not exactly the expected chain is refused: other ops or axes,
a tensor with a second consumer, a convolution that already has a bias, grouped
or dilated convolutions, `output_padding`. Tests mutate the real graph in each
of these ways.

**Why.** The `Add` between a convolution and its `Relu` is the pattern that cost
INT8 accuracy in the first design of the normalization-folded image models
(see [models.md](models.md#normalization-folded-models)): the quantizer has to
quantize the convolution output, then the sum, then the ReLU output, instead of
giving the convolution's output the ReLU's range. The per-layer
`Squeeze`/`Unsqueeze` pairs are layout work between every two layers; the
per-layer stand-in was also the one layout whose NPU outputs at batch 4 and 8
differed from batch 1.

| | Imported (tf2onnx) | Cleaned |
|---|---|---|
| Nodes | 26 | 11: `Reshape`, (`Conv`\|`ConvTranspose` → `Relu`) ×4, `Conv`, `Reshape` |
| FP32 file, b1 | 48,248 bytes (with metadata) | 44,765 bytes |
| Nodes on the CPU EP after ONNX Runtime's fusions | 26 | 9 |
| Cleaned vs imported, all 2,676 validation windows, CPU EP | | **bit-identical** (max abs diff 0) at b1, b4 and b8 |
| Cleaned vs the bundle's references (8 windows) | | 5.25e-6 vs Colab's x64 ONNX Runtime, 5.72e-6 vs Keras |

The rewrite is exact in real arithmetic, and on the CPU EP it is exact in
float32 too: the bias is added once to the finished sum either way. The 5e-6
against the bundle's references is ARM64 versus x64 ONNX Runtime (the imported
graph gives exactly the cleaned graph's outputs), not the clean-up.

## Variants and lineage

All files are made by
[`scripts/prepare_autoencoder.py`](../scripts/prepare_autoencoder.py) into
`models/conv1d_autoencoder/`, each with a check sidecar (`<name>.json`), a
recipe sidecar (`<name>.recipe.json`) and an entry in `models/manifest.json`:

| Variant | What | Parent | b1 size |
|---|---|---|---|
| `fp32_imported` | the source graph unchanged, plus `edgebench.*` metadata (kept as a reference, like the unpruned INT8 files) | none: the root; its weights are the pinned source file | 48,248 |
| `fp32` | the cleaned graph | `fp32_imported` | 44,765 |
| `int8` | the default INT8 recipe on the cleaned graph, pruned | `fp32` | 28,249 |
| `int8_unpruned` | the quantizer's output before pruning (reference) | `fp32` | 28,570 |

The lineage is source file → imported copy → cleaned FP32 → INT8. The imported
copy's graph is the source file byte for byte (checked on every run), and its
recipe's weights block names the source path, its SHA256, the notebook, both
commits and the training record's SHA256; `model_recipes.py --check` compares
these with the registry pins.

The metadata records the model, variant, precision, batch size, the source
(`weights_source colab_bundle`), normalization `external` with the series mean
and std and the split manifest's SHA256 in `edgebench.input_preprocessing`
(window 128, stride 1, layout `[batch, time step, channel]`), the input range
and shape, the training tool versions from the training record (TensorFlow,
Keras, tf2onnx, onnx) and the local onnx version that wrote the files, and for
the cleaned graph and its INT8 files what the clean-up rewrote. There is **no
normalization-folded variant**, and the metadata says why
(`edgebench.fold_not_applicable`): the constants are statistics of this one
series' training split, not a property of the input format the way ImageNet
mean/std are for [0, 1] pixels, so callers normalize anyway, and there is no
fixed raw input range to fold to.

`prepare_autoencoder.py --verify` builds every file again in memory and
compares it with the manifest and the files on disk: all 12 are byte-identical.

## INT8

The default recipe ([models.md](models.md#int8-qdq-models)): signed int8
symmetric per-channel weights, uint8 asymmetric activations, int32 biases,
MinMax calibration on batch 1 over the 200 calibration windows of the split
manifest, in manifest order; batch 4 and 8 reuse the batch-1 ranges, so every
batch size has identical scales and zero points (checked). Pruning finds no dead
Q/DQ chains (there is no Clip), only 4 stale `value_info` entries; the pruned
and unpruned files give bit-identical outputs. On the CPU EP, batch 4 and 8
give batch 1's outputs bit for bit, for FP32 and INT8.

The recipe was also run once on the uncleaned graph, for comparison (not
shipped). On all validation windows, CPU EP, batch 1:

| | INT8 nodes | QuantizeLinear | Mean window MSE (vs input) | Max abs diff vs FP32 |
|---|---|---|---|---|
| FP32 (cleaned) | | | 3.07e-5 | |
| INT8 from the uncleaned graph | 74 | 21 | 1.08e-3 | 0.248 |
| INT8 from the cleaned graph | 31 | 7 | **3.95e-4** | 0.275 |

With the bias inside the convolution, every `Relu` folds into its
convolution's output range and disappears from the INT8 graph (7 quantization
points: the input, the four ReLU-range convolution outputs, the last
convolution's output and the graph output). The clean-up cuts the INT8 reconstruction error by 2.7x. INT8 still
reconstructs about 13 times worse than FP32: the series is smooth and the
errors are small, so the quantization steps of the activations (0.012 to 0.027
in normalized units) are of the same order as FP32's whole reconstruction
error.

## NPU

`scripts/check_autoencoder_npu.py --models` (no timing) opens every INT8 file
(pruned and unpruned) at batch 1, 4 and 8, and the cleaned FP32 file with FP16
compute (`enable_htp_fp16_precision=1`), each in a QNN HTP session with CPU
fallback disabled, parses the verbose placement log and runs all 2,676
validation windows.

| Files | Strict session | Placement | QNN-validated nodes | Rejected | NPU vs CPU (same file), max abs diff | Repeat in session | b4 / b8 vs b1 on the NPU |
|---|---|---|---|---|---|---|---|
| `int8`, `int8_unpruned`, b1/b4/b8 | all 6 created | 1 fused node, all on QNN | 39 (Conv 3, ConvTranspose 2, Reshape 2, Transpose 2, Q 1, DQ 1 units) | 0 | 0.082 | bit-identical | **bit-identical** (0 of 2,676 windows differ) |
| `fp32` with FP16 compute, b1/b4/b8 | all 3 created | 1 fused node, all on QNN | 13 (adds Relu 4) | 0 | 0.0060 (vs CPU FP32) | bit-identical | **bit-identical** |

ONNX Runtime 1.32.0, onnxruntime-qnn 2.8.0, QNN backend 2.52.0.261104115025; AC
power, Balanced power mode. The QNN EP adds a layout `Transpose` at each end
itself. Unlike the image models' INT8 files at batch 8, and unlike the
per-layer stand-in, the cleaned autoencoder computes the same outputs on the
NPU at every batch size, so batch 8 can be used for INT8 NPU runs.

## References and the C++ match

`scripts/make_references.py` stores, for all 12 files, Python ONNX Runtime's
outputs on the 8 reference windows of the split manifest (normalized; input set
`windows-reference-b97ee8e2`) and on the engine's SplitMix64 input
(`splitmix64-seed0-<shape>`), on the CPU, and on the NPU (strict) for the INT8
files. NPU repeats are bit-identical; the NPU differs from the CPU INT8 by at
most 0.055 on the reference windows. The C++ engine's reference test
(`ReferenceModels.*`, ctest label `models`) runs each file through the CPU
provider in the timed path (IoBinding with preallocated buffers, 4 threads):
**all 12 files match Python bit for bit** (9/9 runs at b1, 3/3 at b4, 2/2 at b8).
A short `edgebench run` on each file (2 trials of 50 iterations) succeeds,
validates with `--strict` and places every node on the CPU EP.

## Anomaly-detection evaluation

[`scripts/eval_anomaly.py`](../scripts/eval_anomaly.py); results in
[`benchmarks/results/published/accuracy/anomaly/`](../benchmarks/results/published/accuracy/anomaly/),
schema [`schemas/anomaly-result.v1.0.0.schema.json`](../schemas/anomaly-result.v1.0.0.schema.json)
(see [accuracy.md](accuracy.md#anomaly-detection) for the schema and the
checks).

**Score.** The mean squared reconstruction error of a window (float64 from the
float32 output). Windows are the split manifest's, stride 1.

**Threshold rule: the maximum validation score.** The validation split is
anomaly-free by construction and is never used for training, so its largest
score is the lowest threshold that flags none of the held-out normal data. The
rule needs no percentile choice, and with 2,676 heavily overlapping validation
windows from 2,803 points a high percentile such as p99.9 would sit within a
few windows of the maximum anyway. A test window is flagged when its score is
strictly greater. The threshold is computed per run from that run's own
validation scores, and the test split is never looked at (a test checks that
changing the test scores leaves the threshold alone). Its cost: one unusual
validation window sets the threshold for everything. FP32's validation maximum
is 4x its p99.

**Metrics on the test split** (11,230 windows, 2,649 of them touching a
label): window-level ROC-AUC (Mann-Whitney, ties as one half) and PR-AUC
(average precision, step sum); precision, recall and F1 at the threshold; and
event-level detection: for each of the 4 labelled anomaly windows, whether any
overlapping test window is flagged, and the delay from the start of the
labelled window to the last point of the first flagged window, which is when
a flag could be raised.

**Runs.** CPU FP32 and CPU INT8 (CPU EP, 4 threads), NPU INT8 (strict) and NPU
FP16 compute on the FP32 file (strict). Batch size: for each run, b1, b4 and b8
reconstruct every validation window on that device, and the largest batch size
whose reconstructions equal b1's bit for bit is used. That was b8 for all four
runs. A rerun reproduces every number and the SHA256 of every score file.

| Run | Batch | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | False-positive rate | Events detected |
|---|---|---|---|---|---|---|---|---|---|
| CPU FP32 | 8 | 3.46e-4 | 0.709 | 0.427 | 0.923 | 0.059 | 0.111 | 0.15% | 2 of 4 |
| CPU INT8 | 8 | 1.30e-3 | 0.813 | 0.714 | 0.796 | 0.528 | 0.635 | 4.2% | 4 of 4 |
| NPU INT8 | 8 | 1.14e-3 | 0.805 | 0.706 | 0.796 | 0.531 | 0.637 | 4.2% | 4 of 4 |
| NPU FP16 | 8 | 3.47e-4 | 0.713 | 0.432 | 0.928 | 0.059 | 0.110 | 0.14% | 2 of 4 |

Positives are 23.6% of the test windows, so a random score has PR-AUC 0.236.

Per event (delay from the start of the labelled window, which is 567 points,
47 h 15 min long; 5 minutes per point):

| Labelled window (index range) | CPU FP32 | CPU INT8 | NPU INT8 | NPU FP16 |
|---|---|---|---|---|
| [2126, 2693), from 2013-12-10 06:25 | not detected | 19 points (1 h 35 min) | 19 (1 h 35 min) | not detected |
| [3703, 4270), from 2013-12-15 17:50 | 155 (12 h 55 min) | 155 (12 h 55 min) | 153 (12 h 45 min) | 155 (12 h 55 min) |
| [16057, 16624), from 2014-01-27 14:20 | not detected | 281 (23 h 25 min) | 281 (23 h 25 min) | not detected |
| [19232, 19799), from 2014-02-07 14:55 | 660 (55 h, after the labelled window ended; 8 of 694 windows flagged) | 45 (3 h 45 min) | 41 (3 h 25 min) | 660 (55 h) |

Paired comparisons (`comparisons.json`; each run flags with its own threshold):

| Comparison | Spearman (test scores) | Flag agreement | Flagged a / b | Only a / only b correct | McNemar p |
|---|---|---|---|---|---|
| NPU INT8 vs CPU INT8 | 0.779 | 99.92% | 1,766 / 1,757 | 8 / 1 | 0.039 |
| CPU INT8 vs CPU FP32 | 0.334 | 85.75% | 1,757 / 169 | 1,248 / 352 | 2e-117 |
| NPU FP16 vs CPU FP32 | 0.9993 | 99.98% | 167 / 169 | 1 / 1 | 1.0 |

**What this shows.**

- **The NPU runs the INT8 detector like the CPU does.** Both flag the same
  windows except 9 (all flagged only by the NPU, 8 of them correctly), detect
  all 4 events, and differ by 0 to 4 points in detection time. The scores
  themselves correlate only 0.78 because on normal windows they are dominated
  by INT8 rounding, which the NPU and the CPU do differently (max output
  difference 0.08). The McNemar p of 0.039 is not evidence of a real
  difference: stride-1 windows overlap by 127 of 128 points, so the 9
  discordant windows are a handful of independent events at most, and the
  test's independence assumption does not hold.
- **FP16 on the NPU matches FP32 on the CPU** (Spearman 0.9993, 2 windows
  differ, same events).
- **INT8 detects better than FP32, for a reason that has nothing to do with
  better modelling.** The INT8 input's quantization range comes from the
  calibration windows: [−4.67, 2.29] in normalized units, raw values 52.3 to
  108.5.
  The training data barely leaves it, but 7.6% of the test points do (the
  series drops to −10.9 normalized during the failures), and 54% of the
  anomalous test windows contain such points against 4.4% of the normal ones.
  Clipped inputs cannot be reconstructed, so their error is large: the
  calibration range acts as a range check. FP32 reconstructs the out-of-range
  values well enough that only 2 of the 4 events rise above its threshold, and
  that threshold is set by a single unusual validation window. This is a
  property of this series and this calibration, not something to expect of
  INT8 in general.
- **FP32 is a weak detector here.** Its precision is high (0.92) but it
  flags 6% of the labelled windows; ROC-AUC 0.71 shows the scores do separate
  the classes somewhat. Nothing was retrained or tuned to improve this: the
  model is a benchmark workload, and these are its numbers.

## Limitations

- **One series, four labelled anomalies.** Event-level results are 4 events;
  window-level counts are large only because windows overlap.
- **Overlapping windows are not independent.** Window-level metrics weigh long
  anomalies more, and the McNemar p-values are optimistic. No confidence
  intervals are given for the same reason.
- **The threshold rule is one choice.** The maximum validation score is
  sensitive to a single unusual validation window; another rule (a percentile,
  a margin) would move precision and recall. The rule was fixed before looking
  at test results and is applied the same way to every run.
- **NAB's label windows are wide** (about 47 hours, centred on the recorded
  anomaly), so the window-level positives include many points that look normal,
  and the delay to the start of a labelled window is not a delay to the fault.
- **Training is not bit-reproducible across machines**; the committed bundle is
  the model, and a retrained model would give different numbers.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\get_data.py --dataset nab
.\.venv\Scripts\python.exe scripts\prepare_autoencoder.py            # model files, sidecars, manifest
.\.venv\Scripts\python.exe scripts\prepare_autoencoder.py --verify   # rebuild in memory and compare
.\.venv\Scripts\python.exe scripts\check_autoencoder_npu.py --models # strict NPU sessions, batch equivalence
.\.venv\Scripts\python.exe scripts\make_references.py                # references of all model files
.\.venv\Scripts\python.exe scripts\eval_anomaly.py                   # results in benchmarks/results/published/accuracy/anomaly/
.\.venv\Scripts\python.exe scripts\validate_accuracy.py benchmarks\results\published\accuracy
```
