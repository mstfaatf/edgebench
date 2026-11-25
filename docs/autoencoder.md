# Time-series anomaly-detection autoencoder

> **Status: in progress.** The data, windows, splits, the NPU op-support check
> and the training notebook exist. The trained model, its INT8 variants,
> reference outputs and the anomaly-detection evaluation on CPU and NPU do not
> exist yet; sections that depend on them say so.

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
bit. The exported files and their SHA256 are what the rest of the pipeline
uses. The bundle goes to `models/colab/` (not committed).

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
image models' INT8 graphs at batch 8, its b4/b8 outputs differ from b1. The
exported files will show which layout tf2onnx actually produced, and the
pipeline's NPU check runs again on the real files.

To repeat the check (writes to `benchmarks/results/local/autoencoder/`):

```powershell
.\.venv\Scripts\python.exe scripts\check_autoencoder_npu.py
```

## Still to do

- Import the Colab bundle: registry entry and metadata, recipe sidecar and
  manifest entry, INT8 with the default recipe, NPU strict check on the
  real files, reference outputs and the C++ bit-exact match test.
- Anomaly-detection evaluation on CPU and NPU: threshold from the validation
  reconstruction errors, detection on the test split.
