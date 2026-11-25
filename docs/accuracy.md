# Accuracy

Accuracy of every model variant on ImageNetV2 (matched frequency, the 10,000
image `eval-full` split), on the CPU and on the NPU, so that every latency number
can be read next to the accuracy of the same file on the same device. The
results are deterministic (a rerun gives identical predictions) and are
published in [`benchmarks/results/published/accuracy/`](../benchmarks/results/published/accuracy/).
The time-series autoencoder is evaluated as an anomaly detector instead
([Anomaly detection](#anomaly-detection)).

## What was measured

| Device | Files (pruned) | Session |
|---|---|---|
| CPU | FP32, FP32 folded, INT8, INT8 folded of each model | ONNX Runtime 1.32.0 CPU EP, 4 intra-op threads, sequential, all graph optimizations |
| NPU | INT8, INT8 folded | QNN HTP (QNN EP 2.8.0, QNN 2.52.0), strict mode: `session.disable_cpu_ep_fallback = 1`, so session creation fails if any node would run on the CPU; burst performance mode |
| NPU, FP16 compute | the FP32 (external normalization) file | as above, plus `enable_htp_fp16_precision = 1`: the FP32 file is computed in **FP16** on the HTP. This is not FP32 on the NPU |

21 runs: 12 on the CPU, 9 on the NPU. Every strict NPU session was created; no
model was refused. The unpruned INT8 files give bit-identical outputs to the
pruned ones ([models.md](models.md#pruning)) and are not evaluated separately.

**Images and labels.** The eval-full manifest
(`datasets/imagenetv2-matched-frequency/eval-full.tsv`), in manifest order.
Every label is the manifest's, checked to equal the image's integer folder name
(a string sort of the folder names mislabels 988 of the 1000 classes) and to
index [`scripts/common/imagenet_classes.json`](../scripts/common/imagenet_classes.json)
([setup.md](setup.md#1-download-the-datasets)).

**Preprocessing.** The shared transform (`scripts/common/preprocess.py`): resize
the shorter side to 256 (bilinear), center crop 224, scale to [0, 1]; then the
model's own mean/std for external-normalization files, nothing more for folded
files (which take the [0, 1] tensor). The resized and cropped images are cached
once as uint8 (`data/cache/imagenetv2/`, not committed); converting a cached
image gives exactly the tensor `load_image` gives, which is checked on 200
images before every evaluation.

**Batch size.** Accuracy must not depend on the batch size, so each run first
evaluates eval-quick (1,000 images) at batch 1, 4 and 8 on its own device and
uses the largest batch size whose top-5 predictions equal batch 1's on every
image. On the CPU all three agree, and batch 8 is used. On the NPU:

- **INT8 at batch 4 gives logits bit-identical to batch 1** (all 1,000
  images, MobileNetV2 and ResNet-18 checked directly), so INT8 NPU runs use
  batch 4;
- **INT8 at batch 8 computes different outputs**: every image's logits
  differ from batch 1 (up to 0.06 for MobileNetV2 and 0.17 for ResNet-18), and
  the top-1 class changes on 6 and 10 of the 1,000 images. The HTP evidently
  executes the batch-8 graph differently. Batch 8 is therefore not used for INT8
  on the NPU, and an NPU INT8 benchmark at batch 8 measures a slightly different
  computation from batch 1;
- FP16 at batch 8 matches batch 1, so FP16 runs use batch 8.

Each result records the candidates and the batch size used
(`execution.batch_equivalence`).

**Metrics.** Top-1 and top-5 accuracy with 95% **Wilson score** intervals.
Paired comparisons of two runs on the same images: top-1 agreement (same
predicted class) and **McNemar's exact test** (two-sided binomial test on the
images one run gets right and the other wrong). Per-image top-5 class indices
and logits of every run are kept as `.npy` files
(`benchmarks/results/local/accuracy/predictions/`, not committed); every result
records their SHA256, so any number here can be recomputed from them.

## Results

ImageNetV2 eval-full, 10,000 images.

| Model | File | Device | Compute | Batch | Top-1 % (95% CI) | Top-5 % (95% CI) |
|---|---|---|---|---|---|---|
| MobileNetV2 | FP32 | CPU | FP32 | 8 | 59.56 (58.59-60.52) | 81.89 (81.12-82.63) |
| MobileNetV2 | FP32 folded | CPU | FP32 | 8 | 59.56 (58.59-60.52) | 81.89 (81.12-82.63) |
| MobileNetV2 | INT8 | CPU | INT8 | 8 | 59.18 (58.21-60.14) | 81.84 (81.07-82.58) |
| MobileNetV2 | INT8 folded | CPU | INT8 | 8 | 59.19 (58.22-60.15) | 81.83 (81.06-82.57) |
| MobileNetV2 | INT8 | NPU (strict) | INT8 | 4 | 59.13 (58.16-60.09) | 81.80 (81.03-82.54) |
| MobileNetV2 | INT8 folded | NPU (strict) | INT8 | 4 | 59.33 (58.36-60.29) | 81.72 (80.95-82.47) |
| MobileNetV2 | FP32 file | NPU (strict) | **FP16** | 8 | 59.50 (58.53-60.46) | 81.91 (81.14-82.65) |
| ResNet-18 | FP32 | CPU | FP32 | 8 | 57.28 (56.31-58.25) | 79.89 (79.09-80.66) |
| ResNet-18 | FP32 folded | CPU | FP32 | 8 | 57.28 (56.31-58.25) | 79.89 (79.09-80.66) |
| ResNet-18 | INT8 | CPU | INT8 | 8 | 57.08 (56.11-58.05) | 79.85 (79.05-80.62) |
| ResNet-18 | INT8 folded | CPU | INT8 | 8 | 57.12 (56.15-58.09) | 79.83 (79.03-80.60) |
| ResNet-18 | INT8 | NPU (strict) | INT8 | 4 | 57.21 (56.24-58.18) | 79.81 (79.01-80.59) |
| ResNet-18 | INT8 folded | NPU (strict) | INT8 | 4 | 57.27 (56.30-58.24) | 79.86 (79.06-80.63) |
| ResNet-18 | FP32 file | NPU (strict) | **FP16** | 8 | 57.29 (56.32-58.26) | 79.88 (79.08-80.65) |
| EfficientNet-Lite0 | FP32 | CPU | FP32 | 8 | 62.77 (61.82-63.71) | 84.02 (83.29-84.73) |
| EfficientNet-Lite0 | FP32 folded | CPU | FP32 | 8 | 62.77 (61.82-63.71) | 84.02 (83.29-84.73) |
| EfficientNet-Lite0 | INT8 | CPU | INT8 | 8 | 62.59 (61.64-63.53) | 84.17 (83.44-84.87) |
| EfficientNet-Lite0 | INT8 folded | CPU | INT8 | 8 | 62.72 (61.77-63.66) | 84.14 (83.41-84.84) |
| EfficientNet-Lite0 | INT8 | NPU (strict) | INT8 | 4 | 62.60 (61.65-63.54) | 84.12 (83.39-84.82) |
| EfficientNet-Lite0 | INT8 folded | NPU (strict) | INT8 | 4 | 62.85 (61.90-63.79) | 84.08 (83.35-84.78) |
| EfficientNet-Lite0 | FP32 file | NPU (strict) | **FP16** | 8 | 62.78 (61.83-63.72) | 84.03 (83.30-84.73) |

## Paired comparisons

| Comparison (A vs B) | Top-1 A - B (points) | Top-1 agreement | A right, B wrong | B right, A wrong | McNemar p |
|---|---|---|---|---|---|
| EfficientNet-Lite0: NPU INT8 vs CPU INT8 | +0.01 | 97.09% | 58 | 57 | 1 |
| EfficientNet-Lite0: NPU INT8 folded vs CPU INT8 folded | +0.13 | 97.18% | 70 | 57 | 0.287 |
| EfficientNet-Lite0: CPU INT8 vs CPU FP32 | -0.18 | 94.74% | 105 | 123 | 0.26 |
| EfficientNet-Lite0: CPU INT8 folded vs CPU FP32 folded | -0.05 | 93.95% | 126 | 131 | 0.803 |
| EfficientNet-Lite0: CPU FP32 folded vs CPU FP32 | +0.00 | 100.00% | 0 | 0 | 1 |
| EfficientNet-Lite0: CPU INT8 folded vs CPU INT8 | +0.13 | 95.95% | 91 | 78 | 0.356 |
| EfficientNet-Lite0: NPU FP16 vs CPU FP32 | +0.01 | 99.74% | 7 | 6 | 1 |
| MobileNetV2: NPU INT8 vs CPU INT8 | -0.05 | 95.22% | 97 | 102 | 0.777 |
| MobileNetV2: NPU INT8 folded vs CPU INT8 folded | +0.14 | 95.38% | 111 | 97 | 0.367 |
| MobileNetV2: CPU INT8 vs CPU FP32 | -0.38 | 92.24% | 147 | 185 | 0.0421 |
| MobileNetV2: CPU INT8 folded vs CPU FP32 folded | -0.37 | 92.12% | 157 | 194 | 0.0545 |
| MobileNetV2: CPU FP32 folded vs CPU FP32 | +0.00 | 100.00% | 0 | 0 | 1 |
| MobileNetV2: CPU INT8 folded vs CPU INT8 | +0.01 | 94.34% | 118 | 117 | 1 |
| MobileNetV2: NPU FP16 vs CPU FP32 | -0.06 | 99.65% | 4 | 10 | 0.18 |
| ResNet-18: NPU INT8 vs CPU INT8 | +0.13 | 96.26% | 68 | 55 | 0.279 |
| ResNet-18: NPU INT8 folded vs CPU INT8 folded | +0.15 | 96.15% | 74 | 59 | 0.225 |
| ResNet-18: CPU INT8 vs CPU FP32 | -0.20 | 95.05% | 83 | 103 | 0.163 |
| ResNet-18: CPU INT8 folded vs CPU FP32 folded | -0.16 | 94.66% | 93 | 109 | 0.291 |
| ResNet-18: CPU FP32 folded vs CPU FP32 | +0.00 | 100.00% | 0 | 0 | 1 |
| ResNet-18: CPU INT8 folded vs CPU INT8 | +0.04 | 95.04% | 94 | 90 | 0.825 |
| ResNet-18: NPU FP16 vs CPU FP32 | +0.01 | 99.89% | 3 | 2 | 1 |

What they say:

- **The NPU matches the CPU on the same INT8 file.** For every model and both
  normalizations, NPU INT8 top-1 is within 0.15 points of CPU INT8 (from -0.05
  to +0.15), and no difference is statistically significant (McNemar p = 0.22
  to 1). The two devices give the same top-1 class on 95.2-97.2% of images:
  their outputs are not identical (different kernels and accumulation), but
  the disagreements cancel out in accuracy.
- **INT8 costs little.** CPU INT8 is 0.05-0.38 points below CPU FP32. Only
  MobileNetV2 external normalization reaches p < 0.05 (-0.38 points,
  p = 0.042); its folded counterpart is borderline (-0.37, p = 0.055).
  INT8 and FP32 agree on 92-95% of top-1 predictions.
- **Folding the normalization is free.** FP32 folded and FP32 external give
  the same top-1 class on all 10,000 images. INT8 folded vs INT8 external
  differ by at most 0.13 points (not significant): they are separately
  calibrated files, so their predictions differ on 4-6% of images.
- **FP16 on the NPU is as accurate as FP32 on the CPU**: within 0.06 points,
  same top-1 class on 99.65-99.89% of images, not significant.

## Reproducing

```powershell
# needs the exported, folded and quantized models, ImageNetV2 (get_data.py) and the QNN EP
.\.venv\Scripts\python.exe scripts\eval_accuracy.py            # about 15 minutes
.\.venv\Scripts\python.exe scripts\validate_accuracy.py benchmarks\results\published\accuracy
```

`--only RUN_ID ...` reruns selected runs (with `--output-dir` and
`--predictions-dir` to keep the published files). Reruns of a CPU FP32, an NPU
INT8 and an NPU FP16 run gave identical metrics and byte-identical prediction
files. The results follow
[`schemas/accuracy-result.v1.0.0.schema.json`](../schemas/accuracy-result.v1.0.0.schema.json);
each records the model file's SHA256 (as in `models/manifest.json` and its
recipe sidecar), the device, execution provider, strict mode, compute precision,
batch size, the dataset manifest's SHA256, the preprocessing and the
environment versions. `validate_accuracy.py` (standard library and jsonschema;
also run in CI) checks the schema and that every number is consistent: accuracy
= correct / total, the Wilson intervals, the comparison counts against both
runs, and the McNemar p-values.

## Caveats

- **ImageNetV2 is harder than ImageNet val by design.** It was collected to
  test generalization, and models score lower on it: here about 12.5 points
  below the ImageNet-1k val top-1 torchvision reports for MobileNetV2 (72.15)
  and ResNet-18 (69.76). Compare variants with each other, not with published
  ImageNet numbers.
- **One fixed preprocessing for every model** (resize 256, crop 224,
  bilinear). Some weights were evaluated by their authors with other settings:
  torchvision evaluates MobileNetV2's IMAGENET1K_V2 weights with resize 232,
  and timm evaluates EfficientNet-Lite0 with bicubic interpolation. The
  absolute numbers may differ slightly from the recommended configuration;
  comparisons between variants and devices are unaffected.
- **NPU FP16 is FP16 compute.** It runs the FP32 file with
  `enable_htp_fp16_precision = 1`; FP32 compute on the NPU was not evaluated.
- **The INT8 calibration set is Imagenette** (200 images from the ImageNet
  train distribution), disjoint from these ImageNetV2 images
  ([models.md](models.md#int8-qdq-models)).
- Accuracy is measured with Python's ONNX Runtime. The C++ engine computes
  bit-identical outputs for the same files on the CPU
  ([models.md](models.md#reference-outputs)), so these numbers hold for it too.

## Anomaly detection

The Conv1D autoencoder ([autoencoder.md](autoencoder.md)) is evaluated on the
test split of NAB `machine_temperature_system_failure` by
`scripts/eval_anomaly.py`: the score is each window's reconstruction MSE, the
threshold is the largest score of any validation window (chosen on the
validation split only), and the metrics are window-level ROC-AUC, PR-AUC and
precision / recall / F1 at the threshold, plus how many of the 4 labelled
anomalies get at least one flagged window, and how early. Each run uses the
largest batch size whose reconstructions equal batch 1's bit for bit on every
validation window on that device (b8 for all four).

| Run | ROC-AUC | PR-AUC | F1 | Events detected |
|---|---|---|---|---|
| CPU FP32 | 0.709 | 0.427 | 0.111 | 2 of 4 |
| CPU INT8 | 0.813 | 0.714 | 0.635 | 4 of 4 |
| NPU INT8 (strict) | 0.805 | 0.706 | 0.637 | 4 of 4 |
| NPU FP16 (strict, FP32 file) | 0.713 | 0.432 | 0.110 | 2 of 4 |

NPU INT8 and CPU INT8 flag the same test windows except 9 (99.92% agreement)
and detect the same events within 0 to 4 points; NPU FP16 matches CPU FP32 on
all but 2 windows. INT8 detects more than FP32 because its input quantization
range, calibrated on training windows, clips the out-of-range values of the
failures; see [autoencoder.md](autoencoder.md#anomaly-detection-evaluation)
for the per-event delays, the comparisons and the limitations (one series, 4
events, overlapping windows).

**Why a separate schema.** The classification results' blocks (top-1/top-5
with Wilson intervals, image preprocessing, an ImageNetV2 dataset block,
top-5 prediction files) do not describe a detector. An additive version of
the accuracy schema with a task type would have turned each of those blocks
into a per-task choice and weakened both. Anomaly results therefore have their
own schema,
[`schemas/anomaly-result.v1.0.0.schema.json`](../schemas/anomaly-result.v1.0.0.schema.json),
version key `anomaly_schema_version`, with the same structure (kinds `run`
and `comparisons`, model file SHA256, execution block, batch equivalence,
environment) and the detector's own blocks: dataset and split manifest,
input normalization, score definition, threshold and its rule, test metrics
with per-event detection, and the SHA256 of the per-window score files (`.npy`,
in `benchmarks/results/local/accuracy/anomaly_scores/`, not committed). The
results live in their own directory,
[`benchmarks/results/published/accuracy/anomaly/`](../benchmarks/results/published/accuracy/anomaly/),
so the classification checks ("one dataset manifest per directory") stay as
they are. `scripts/validate_accuracy.py` validates both kinds (it picks the
schema by version key; CI runs it on the whole `accuracy/` tree): for anomaly
runs, that the confusion counts add up, precision / recall / F1 follow from
them, no validation window exceeds the threshold, and each event's detection
point and delay follow from its first flagged window; for comparisons, that
the counts match both runs, the discordant windows are the windows flagged by
only one run, and the McNemar p-values are recomputed. A test recomputes every
metric from the score files when they are present.

```powershell
# needs the autoencoder files (prepare_autoencoder.py), the NAB data (get_data.py --dataset nab) and the QNN EP
.\.venv\Scripts\python.exe scripts\eval_anomaly.py              # under a minute
.\.venv\Scripts\python.exe scripts\validate_accuracy.py benchmarks\results\published\accuracy
```
