# Models

EdgeBench benchmarks ImageNet classifiers exported from PyTorch (torchvision and
timm) to ONNX with fixed shapes, one file per batch size, in FP32 and as static
INT8 QDQ models. This page describes which models, where their weights come
from, how they are exported, quantized and checked, and what input they expect.
It also benchmarks a time-series anomaly-detection autoencoder trained with
Keras in Colab and imported from committed ONNX files
([Imported models](#imported-models); details in [autoencoder.md](autoencoder.md)).

## Supported models

| Model | Name | Weights | Weights file (SHA256) | License |
|---|---|---|---|---|
| MobileNetV2 | `mobilenetv2` | torchvision `MobileNet_V2_Weights.IMAGENET1K_V2` (ImageNet-1k top-1 72.154 per torchvision) | [`mobilenet_v2-7ebf99e0.pth`](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth) (`7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`, 14,258,573 bytes) | torchvision BSD-3-Clause; weights trained on ImageNet |
| ResNet-18 | `resnet18` | torchvision `ResNet18_Weights.IMAGENET1K_V1` (69.758 per torchvision) | [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) (`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, 46,830,571 bytes) | torchvision BSD-3-Clause; weights trained on ImageNet |
| EfficientNet-Lite0 | `efficientnet_lite0` | timm `efficientnet_lite0.ra_in1k`, Hugging Face [`timm/efficientnet_lite0.ra_in1k`](https://huggingface.co/timm/efficientnet_lite0.ra_in1k) at revision `29046ba6cc39eacfd5c7e6ffc1ae6188cfaa7609` | `model.safetensors` (`96c33bc4ce15e6f834b0f7da41d4bae33547fff1e88e0ba1e812a148693d5dd4`, 18,802,812 bytes) | Apache-2.0 (model card); timm Apache-2.0 |

- MobileNetV2 uses `IMAGENET1K_V2`, torchvision's improved training recipe.
- ResNet-18 uses `IMAGENET1K_V1`, the only ImageNet weights torchvision
  provides for it (the original training recipe), so the two models were not
  trained the same way.
- EfficientNet-Lite0 uses weights trained by timm (RandAugment recipe); see
  [Choosing the EfficientNet-Lite0 weights](#choosing-the-efficientnet-lite0-weights).
- torchvision weights are downloaded by torchvision into its cache
  (`%USERPROFILE%\.cache\torch\hub\checkpoints`). torchvision's file names and its
  download check only use the first 8 hex digits of the SHA256, so the export
  verifies the full SHA256 pinned in `scripts/common/model_registry.py` before
  using a file, and stops if it differs. timm weights are downloaded with
  `huggingface_hub` at the pinned revision into the Hugging Face cache and are
  checked the same way. (On Windows without Developer Mode, `huggingface_hub`
  warns that it cannot create symlinks in its cache; it copies files instead,
  which is harmless.)
- torchvision's top-1 figures are on the ImageNet-1k validation set with
  torchvision's own evaluation transform (for MobileNetV2 `IMAGENET1K_V2` that
  resizes to 232, not 256). EdgeBench evaluates on ImageNetV2 with one shared
  transform, so its numbers are not comparable with these.

The model list is `MODELS` in `scripts/common/model_registry.py`. To add a
model, add a `ModelSpec` there: for torchvision the constructor, weights enum
and member, weights URL, SHA256 and size; for timm the architecture, pretrained
tag, Hugging Face repository, revision and file, SHA256 and size; for both the
mean/std the weights were trained with. Then run the export for it and commit
the updated manifest. The export refuses a model whose registry mean/std differ
from the ones torchvision or timm declare for the weights.

### Choosing the EfficientNet-Lite0 weights

torchvision has no EfficientNet-Lite. timm has two sets of Lite0 weights, both
Apache-2.0 on Hugging Face:

| | `tf_efficientnet_lite0.in1k` | `efficientnet_lite0.ra_in1k` (used) |
|---|---|---|
| Origin | Google's TensorFlow weights (paper authors), ported to PyTorch by timm | Trained in timm, RandAugment recipe |
| timm preprocessing config | mean/std 0.5, bicubic, crop_pct 0.875, 224x224 | ImageNet mean/std, bicubic, crop_pct 0.875, 224x224 |
| Exported graph (batch 1) | 585 nodes: 49 Conv, 43 Clip, 29 Add, 5 Pad, plus 13 Shape, 5 ConstantOfShape, 46 Cast, 10 Ceil, 35 Sub, 18 Div, 10 Mul, 15 Concat, 10 Reshape, 13 Gather, 40 Unsqueeze, 5 Slice, 5 Transpose, 231 Constant | 160 nodes: 49 Conv, 33 Clip, 9 Add, 66 Constant, Gemm, GlobalAveragePool, Flatten; no Pad |
| Fixed shapes | no: TF-style "same" padding is computed from the input shape at run time, and the output batch dimension comes out symbolic | yes |
| eval-quick top-1 (own mean/std) | 61.4% | 62.1% |
| eval-quick top-1 with the other model's mean/std | 54.9% | 56.7% |

The ported TensorFlow weights were the first choice, being the original
release. But TensorFlow's "same" padding for stride-2 convolutions depends on
the input size, and timm implements it in Python from the tensor's shape. The
TorchScript exporter records that as `Shape` -> arithmetic -> `Pad` subgraphs,
which its constant folding does not fold even though the input shape is fixed
(timm's alternative export convolution gives the same graph). ONNX Runtime
folds them at session creation, but the file itself is not static, and the
extra shape arithmetic would sit in the graph handed to the quantizer and to
the NPU. The timm-trained weights use symmetric padding, export to a clean
static graph, and score slightly higher, so EdgeBench uses them.

The last row shows why the normalization constants are per model: using the
wrong mean/std does not fail, it costs 5-7 points of accuracy.

### Imported models

| Model | Name | Source | Task |
|---|---|---|---|
| Conv1D autoencoder | `conv1d_autoencoder` | committed bundle [`models/sources/conv1d_autoencoder/`](../models/sources/conv1d_autoencoder/): FP32 ONNX at batch 1, 4 and 8 exported by tf2onnx 1.17.0 from Keras 3.15.1, trained by `notebooks/train_autoencoder.ipynb` (commit `8d8f5a3`); 9,505 parameters | anomaly detection on NAB `machine_temperature_system_failure` (MIT) |

TensorFlow does not run on Windows on Arm, and Colab training is not
bit-reproducible across machines, so the exported files are the source: they
are committed and pinned by SHA256 in `IMPORTED_MODELS`
(`scripts/common/model_registry.py`), together with the training record, the
notebook commit and the commit it cloned. `scripts/prepare_autoencoder.py`
makes the variants from them: `fp32_imported` (the source graph with
`edgebench.*` metadata, kept as a reference), `fp32` (the source graph after an
exact clean-up: bias `Add`s folded into the convolutions, per-layer
`Squeeze`/`Unsqueeze` removed; `scripts/common/onnx_cleanup.py`), and `int8` /
`int8_unpruned` with the default INT8 recipe, calibrated on the 200 calibration
windows of the split manifest. Normalization is external (the series mean/std
of the training split); there is no folded variant. `--verify` rebuilds every
file in memory and compares it with the manifest. Everything below about
metadata, recipes, the manifest and references applies to these files too;
[autoencoder.md](autoencoder.md) has the clean-up, NPU and evaluation results.

## Export settings

| Setting | Value |
|---|---|
| Exporter | `torch.onnx.export`, TorchScript-based (`dynamo=False`) |
| Opset | 17 (default domain); IR version 8 |
| Precision | FP32 weights and activations |
| Input | `input`, float32 `[batch, 3, 224, 224]`, NCHW, RGB, normalized (see below) |
| Output | `logits`, float32 `[batch, 1000]`, raw logits (no softmax), ImageNet-1k class order (as torchvision) |
| Batch sizes | 1, 4, 8: one file each, batch dimension fixed |
| Dynamic axes | none: every graph input and output has concrete dimensions |
| Graph | constant folding as done by the exporter; no further optimization (ONNX Runtime optimizes at session creation) |
| Versions used | torch 2.13.0+cpu, torchvision 0.28.0, timm 1.0.31, onnx 1.23.2 (recorded in `models/manifest.json`) |

These are the settings the feasibility study validated on the CPU and QNN
execution providers ([feasibility.md](feasibility.md)); the MobileNetV2 batch-1
graph is identical to the one used there, with metadata added.

### Input: normalization is outside the model

The files expect input that is **already normalized**: the output of
`scripts/common/preprocess.py` `load_image(path, mean=..., std=...)`, which
resizes the shorter side to 256 (bilinear), center-crops 224, scales to [0, 1]
and applies the model's own mean/std from the registry. Feeding [0, 1] or 0-255
pixels, or using another model's constants, gives wrong or worse predictions
without any error. Preprocessing is never part of the timed region (see
[methodology.md](methodology.md)).

| Model | Mean / std | Geometry and interpolation | Model's own recommendation |
|---|---|---|---|
| MobileNetV2 | ImageNet `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | resize 256, crop 224, bilinear | torchvision evaluates `IMAGENET1K_V2` with resize **232**, crop 224, bilinear |
| ResNet-18 | ImageNet | resize 256, crop 224, bilinear | same as used |
| EfficientNet-Lite0 | ImageNet | resize 256, crop 224, bilinear | timm: crop_pct 0.875 (= resize 256, crop 224) with **bicubic** interpolation |

The geometry and interpolation are deliberately the same for every model, so
accuracy comparisons between models are like-for-like; only the mean/std
follow each model's training. Where a model's own recommendation differs
(MobileNetV2's 232 resize, EfficientNet-Lite0's bicubic), its accuracy here can
be slightly lower than its publisher's figure.

This is recorded in the file as ONNX metadata: `edgebench.normalization =
external`, and the exact constants in `edgebench.input_preprocessing` (JSON with
`mean` and `std`). `edgebench run` reads the first key and writes it to the
result's `model.normalization`; the results schema has no field for the
constants themselves, so they are read from the model file. A later variant with the
normalization folded into the first convolution will carry `folded`, so the
two can be told apart from the file alone.

### ONNX metadata

Every exported file carries these `metadata_props`. They are the source of
truth for how the file was made; the [recipe sidecar and the
manifest](#recipes-and-the-manifest) are built from them.

| Key | Example |
|---|---|
| `edgebench.model` | `mobilenetv2` |
| `edgebench.variant`, `edgebench.precision` | `fp32`, `fp32` |
| `edgebench.batch_size` | `4` |
| `edgebench.normalization` | `external` |
| `edgebench.input_range` | `(x - mean) / std for x in [0, 1]` (`[0, 1]` for folded files) |
| `edgebench.input_preprocessing` | JSON: resize, crop, scale, mean, std, layout, dtype |
| `edgebench.weights` | `torchvision MobileNet_V2_Weights.IMAGENET1K_V2`, or `timm efficientnet_lite0.ra_in1k (timm/efficientnet_lite0.ra_in1k@<revision>)` |
| `edgebench.weights_source`, `edgebench.weights_url` | `torchvision` or `timm`, the download URL |
| `edgebench.weights_sha256` | SHA256 of the weights file |
| `edgebench.weights_hf_repo`, `.weights_hf_revision`, `.weights_file` (timm models only) | `timm/efficientnet_lite0.ra_in1k`, the pinned revision, `model.safetensors` |
| `edgebench.exporter`, `edgebench.opset`, `edgebench.ir_version` | `torch.onnx.export (TorchScript, dynamo=False)`, `17`, `8` |
| `edgebench.torch_version`, `edgebench.torchvision_version`, `edgebench.onnx_version` | `2.13.0+cpu`, `0.28.0+8fb8771`, `1.23.2` |
| `edgebench.timm_version` (timm models only) | `1.0.31` |

## Files

```
models/
  manifest.json                       committed: index of every file below (hash, lineage)
  mobilenetv2/
    mobilenetv2_fp32_b1.onnx          + .json (checks) + .recipe.json (recipe)
    mobilenetv2_fp32_b4.onnx
    mobilenetv2_fp32_b8.onnx
    mobilenetv2_int8_b1.onnx          + .json: INT8 QDQ, pruned (the benchmarked files)
    mobilenetv2_int8_b4.onnx, ..._b8.onnx
    mobilenetv2_int8_unpruned_b1.onnx   quantizer output before pruning (reference)
    ...
  resnet18/
    resnet18_fp32_b1.onnx
    ...
  efficientnet_lite0/
    efficientnet_lite0_fp32_b1.onnx
    ...
  conv1d_autoencoder/
    conv1d_autoencoder_fp32_imported_b1.onnx   source graph + metadata (reference)
    conv1d_autoencoder_fp32_b1.onnx            cleaned graph
    conv1d_autoencoder_int8_b1.onnx, ..._int8_unpruned_b1.onnx
    ...
  sources/
    conv1d_autoencoder/               committed: the Colab bundle (ONNX, training record,
                                      reference arrays, SHA256SUMS)
```

Files are named `<model>_<variant>_b<batch>.onnx`, with variant `fp32`, `int8`
or `int8_unpruned`, and the normalization-folded `fp32_folded`, `int8_folded`
and `int8_folded_unpruned` ([below](#normalization-folded-models)). The `.onnx` files and
sidecars are not committed (`models/` is gitignored except its README,
`manifest.json` and `sources/`); they travel together. The autoencoder has the
variants `fp32_imported`, `fp32`, `int8` and `int8_unpruned`. Two sidecars sit next to each file:
`<name>.json` records the results of the checks the making script ran (see
below), and `<name>.recipe.json` records how the file was made
([Recipes and the manifest](#recipes-and-the-manifest)). The image models'
unpruned INT8 files have only the recipe sidecar: the checks that involve them
(pruned == unpruned) are recorded in the pruned file's `<name>.json`. The INT8 models of the
feasibility study (`scripts/quantize_qnn.py`) also live in `models/mobilenetv2/`;
they have no recipe sidecar and are not in the manifest.

## Exporting and verifying

```powershell
# all models, batch sizes 1, 4 and 8
.\.venv\Scripts\python.exe scripts\export_models.py

# a subset, e.g. EfficientNet-Lite0 only
.\.venv\Scripts\python.exe scripts\export_models.py --model efficientnet_lite0
.\.venv\Scripts\python.exe scripts\export_models.py --model resnet18 --batch 1 8

# check EfficientNet-Lite0 against the manifest
.\.venv\Scripts\python.exe scripts\export_models.py --verify --model efficientnet_lite0

# re-export in memory and compare with models/manifest.json (writes nothing)
.\.venv\Scripts\python.exe scripts\export_models.py --verify
```

`scripts/export_mobilenetv2.py` is kept as a shortcut for
`export_models.py --model mobilenetv2 --batch 1`.

Exporting is deterministic: exporting again gives byte-identical files, which
`--verify` checks against the committed manifest. That holds for the same
torch, torchvision, timm and onnx versions; with other versions `--verify` says so
and the files are expected to differ. ImageNetV2 (`scripts/get_data.py
--dataset imagenetv2`) is needed for the comparisons with PyTorch; without it
they are skipped with a message.

## Checks

Before a file is written:

- `onnx.checker` with `full_check=True`, and ONNX shape inference in strict
  mode;
- every graph input and output has concrete dimensions, the declared shapes
  are `[batch, 3, 224, 224]` and `[batch, 1000]`, and shape inference gives a
  concrete shape for every intermediate tensor.

After it is written, ONNX Runtime 1.32.0 (CPU EP, 4 threads) runs it on 64
ImageNetV2 eval-quick images (64 different classes), normalized with the
model's own constants:

- **ONNX Runtime vs PyTorch**: the largest absolute logit difference must be at
  most 1e-4, and top-1 must agree on every image;
- **batch consistency**: the batch-4 and batch-8 files must give the same
  logits as the batch-1 file, row by row, within the same tolerance;
- MobileNetV2 batch 1: an input filled with 0.5 (the input `tools/ort_probe`
  uses) must give class 677.

**Why 1e-4.** Both sides compute the same float32 graph, but with different
kernels, operator fusions and batch-dependent blocking, so sums are formed in a
different order. Each reordering changes a result by about one rounding step,
which for float32 logits of magnitude ~20 is ~2e-6; the error grows with the
depth of the network and the length of the dot products (up to 4,608 terms in
ResNet-18). The measured differences are 1e-5 to 2.3e-5 for all three models,
so 1e-4 leaves a 4-9x margin; EfficientNet-Lite0 needs no other value. A real export error (wrong weights, layout or layer) changes logits by
order 1 to 10, four to five orders of magnitude more.

Measured (torch 2.13.0, ONNX Runtime 1.32.0, 64 images):

| File | Max abs diff vs PyTorch | Top-1 agreement | Max abs diff vs batch 1 |
|---|---|---|---|
| `mobilenetv2_fp32_b1` | 1.07e-5 | 100% | – |
| `mobilenetv2_fp32_b4` | 1.12e-5 | 100% | 4.77e-6 |
| `mobilenetv2_fp32_b8` | 1.12e-5 | 100% | 4.77e-6 |
| `resnet18_fp32_b1` | 2.29e-5 | 100% | – |
| `resnet18_fp32_b4` | 2.29e-5 | 100% | 0 |
| `resnet18_fp32_b8` | 2.29e-5 | 100% | 0 |
| `efficientnet_lite0_fp32_b1` | 1.91e-5 | 100% | – |
| `efficientnet_lite0_fp32_b4` | 1.91e-5 | 100% | 5.25e-6 |
| `efficientnet_lite0_fp32_b8` | 1.91e-5 | 100% | 5.25e-6 |

As a sanity check of the export (not a published accuracy figure), top-1 /
top-5 on ImageNetV2 eval-quick (1,000 images, one per class) is 58.7% / 81.2%
for MobileNetV2, 56.4% / 79.8% for ResNet-18 and 62.1% / 84.0% for
EfficientNet-Lite0. A broken export or label
mapping scores near 0.1%.

The tests in `scripts/tests/test_export_models.py` repeat these checks (and
skip with a reason when torch, the weights, the exported files or ImageNetV2
are missing); `OnnxInspectModels.ExportedFp32ModelsHaveFixedShapes` (CTest
label `models`) checks the shapes and metadata with the engine's own ONNX
reader.

## INT8 QDQ models

`scripts/quantize_models.py` turns every FP32 file into a static INT8 model in
QDQ format (QuantizeLinear/DequantizeLinear around FP32 operators), the format
both the ONNX Runtime CPU EP and the QNN HTP (NPU) backend run as integer code.

### Recipe

| Setting | Value |
|---|---|
| Tool | ONNX Runtime 1.32.0 quantization with its QNN helpers: `qnn_preprocess_model`, `get_qnn_qdq_config(per_channel=True, weight_type=QInt8)`, `quantize_static` |
| Weights | signed int8, symmetric (zero point 0), per output channel |
| Activations | uint8, asymmetric, per tensor |
| Biases | int32 |
| Calibration | MinMax over the 200 Imagenette calibration images ([`datasets/imagenette/calibration.tsv`](../datasets/README.md)), in manifest order, preprocessed with the model's own mean/std, one image at a time on the batch-1 model |
| Batch sizes | 4 and 8 reuse the batch-1 calibration ranges (ONNX Runtime's calibration cache), so all batch sizes of a model have identical scales and zero points |
| Pruning | dead QuantizeLinear/DequantizeLinear chains on constants removed (see below) |
| Normalization | unchanged: input normalized outside the model, as for FP32 |

This is the recipe the feasibility study settled on for MobileNetV2 ("s8w"):
per-channel int8 weights agree best with FP32 on the CPU and run fully on the
NPU in strict mode ([feasibility.md](feasibility.md)). The MobileNetV2 batch-1
INT8 graph is identical to that study's `mobilenetv2_int8qdq_pc_s8w_b1.onnx`,
apart from the metadata.

The quantizer reads the FP32 graph without its metadata (ONNX Runtime 1.32's
`qnn_preprocess_model` fails on models that carry `metadata_props`), and the
INT8 metadata is written from the FP32 file afterwards.

### Pruning

For `Clip` (ReLU6) nodes, the quantizer folds the clipping into the activation
ranges but leaves a Constant -> QuantizeLinear -> DequantizeLinear chain on each
min/max constant that nothing reads any more, and ONNX Runtime does not remove
them. `scripts/common/onnx_prune.py` removes nodes whose outputs nothing
consumes (repeatedly), then unused initializers and stale `value_info`. The
unpruned file is kept as `<model>_int8_unpruned_b<batch>.onnx`, and the pruned
file must give bit-identical logits on 64 ImageNetV2 images.

| Model | Nodes before -> after | Removed |
|---|---|---|
| MobileNetV2 | 513 -> 303 | 70 chains (70 Constant, 70 QuantizeLinear, 70 DequantizeLinear) |
| ResNet-18 | 140 -> 140 | none (ReLU, no Clip); 17 stale `value_info` entries |
| EfficientNet-Lite0 | 483 -> 285 | 66 chains |

### Metadata

INT8 files keep the FP32 file's `edgebench.*` keys that describe the model and
its input (model, batch size, weights and their source, `normalization`,
`input_range`, `input_preprocessing`, export versions), and rewrite or add:

| Key | Value |
|---|---|
| `edgebench.variant` | `int8` (`int8_unpruned` for the reference copies) |
| `edgebench.precision` | `int8` |
| `edgebench.opset`, `edgebench.ir_version` | of the INT8 file |
| `edgebench.source_model`, `edgebench.source_sha256` | the FP32 file it was made from |
| `edgebench.onnxruntime_version` | `1.32.0` |
| `edgebench.quant.format`, `.method` | `qdq`, `static` |
| `edgebench.quant.weight_type`, `.weight_symmetric`, `.per_channel` | `int8`, `true`, `true` |
| `edgebench.quant.activation_type`, `.activation_symmetric`, `.bias_type` | `uint8`, `false`, `int32` |
| `edgebench.quant.calibration_method`, `.calibration_dataset`, `.calibration_count`, `.calibration_manifest_sha256`, `.calibration_batch` | `MinMax`, the calibration split, `200`, SHA256 of `calibration.tsv`, `1 (ranges shared by all batch sizes)` |
| `edgebench.quant.tool`, `.qnn_preprocess`, `.pruned` | quantizer and version, whether QNN preprocessing changed the graph, `true`/`false` |
| `edgebench.quant.pruned_chains`, `.pruned_nodes` | dead Q/DQ chains and nodes removed by pruning (`0` in the unpruned files) |

The `edgebench.quant.*` keys are flat strings; the recipe sidecar turns them
into typed fields. `edgebench run` reports these models as `stored_precision: int8`,
`quantization.method: static`, `format: qdq`, int8 weights and uint8
activations (read from the graph), fills `quantization.calibration` (dataset,
sample count) and `quantization.tool` from the metadata, and keeps
`model.normalization: external`.

### Quantizing and verifying

```powershell
# needs the FP32 exports and the Imagenette calibration images (get_data.py)
.\.venv\Scripts\python.exe scripts\quantize_models.py
.\.venv\Scripts\python.exe scripts\quantize_models.py --model resnet18 --batch 1

# quantize again in memory and compare with models/manifest.json (writes nothing)
.\.venv\Scripts\python.exe scripts\quantize_models.py --verify
```

Quantization is deterministic: running it again gives byte-identical files,
pruned and unpruned, which `--verify` checks against the manifest. The script
fails if any scale or zero point differs between the batch sizes of a model, if
a pruned file's logits differ from its unpruned file's, or if batch 4 or 8
differs from batch 1 (1e-4, as for FP32; measured: identical).

### Sanity accuracy (not the published evaluation)

ImageNetV2 eval-quick (1,000 images, one per class), ONNX Runtime CPU EP:

| Model | FP32 top-1 | INT8 top-1 | FP32/INT8 top-1 agreement |
|---|---|---|---|
| MobileNetV2 | 58.7% | 58.9% | 92.6% |
| ResNet-18 | 56.4% | 57.3% | 94.6% |
| EfficientNet-Lite0 | 62.1% | 61.6% | 95.2% |

These only show that quantization did not break the models; with 1,000
images, differences of a point are noise. Published accuracy uses eval-full.

On the NPU (QNN HTP, strict mode: session creation fails if any node would
fall back to the CPU), every INT8 batch-1 model creates a session and runs.
Top-1 agreement with the same INT8 model on the CPU EP, on 64 ImageNetV2
images: MobileNetV2 92.2%, ResNet-18 90.6%, EfficientNet-Lite0 98.4%. The HTP's
integer kernels round differently from ONNX Runtime's CPU kernels, so the two
are not bit-identical.

## Normalization-folded models

The default files take normalized input (`edgebench.normalization = external`).
The folded variants have the mean/std normalization folded into the first
convolution: they take the [0, 1] image tensor straight from resize, crop and
scale, with no normalization step. They show whether moving normalization
into the model's weights is free, compared with doing it outside the model
(not timed) or as separate Sub/Div nodes in the graph (which the feasibility
study found costs about half the NPU cycles of Qualcomm AI Hub's MobileNetV2).

| File | Made by | Input |
|---|---|---|
| `<model>_fp32_folded_b<batch>.onnx` | `scripts/fold_models.py`, from `<model>_fp32_b<batch>.onnx` | [0, 1], no mean/std |
| `<model>_int8_folded_b<batch>.onnx` | `scripts/quantize_models.py --normalization folded`, from the folded FP32 file | [0, 1], no mean/std |
| `<model>_int8_folded_unpruned_b<batch>.onnx` | the same, before pruning (reference) | [0, 1], no mean/std |

**Input contract.** Feed `load_image(path, normalized=False)`: values in
[0, 1], RGB, NCHW. Applying mean/std to the input of a folded model normalizes
twice and gives wrong predictions without any error. The file says so in its
metadata (`edgebench.normalization = folded`, `edgebench.input_range = [0, 1]`),
and `edgebench run` reports `model.normalization: folded`.

### How the fold works: padding with the mean

With raw input `x`, mean `m` and std `s` (per channel), the external model
computes `y = conv(pad0((x - m) / s), W) + b`. Folding gives the usual
`W' = W / s` (per input channel) and `b' = b - sum(W * m / s)`. That alone is
exact only for an unpadded convolution: the original model pads with zeros in
**normalized** space, the folded one would pad with zeros in **raw** space, so
outputs whose kernel touches the padding would differ.

A zero in normalized space is the value `m` in raw space. So the folded model
pads the raw image with the per-channel mean instead, and its convolution pads
nothing:

```
pad0((x - m) / s) = (padm(x) - m) / s
y = conv(padm(x), W') + b'          (Conv pads set to 0)
```

`padm` is two `Concat` nodes in front of the first convolution: constant strips
holding `m` are concatenated above and below the input (axis 2), then left and
right of the result (axis 3). Each strip is exactly as wide as the
convolution's original padding on that side, so asymmetric padding works;
stride and dilation are unchanged, and a side without padding gets no strip.
A first convolution without padding gets no `Concat` at all. The strips are
small (MobileNetV2 and EfficientNet-Lite0: 3x3, stride 2, pad 1, about 11 KB per
image in the batch; ResNet-18: 7x7, stride 2, pad 3, about 33 KB).

Folding is refused, with an error, unless the graph input has exactly one
consumer (a `Conv` with constant weights, `group` 1 and explicit padding) and a
fixed NCHW shape.

**Why padding and not a correction after the convolution.** The border error
of the naive fold does not depend on the input, so it can also be fixed by
adding a precomputed `[1, C_out, H_out, W_out]` correction tensor after the
convolution. That is just as exact in FP32, but it puts an `Add` between the
first convolution and its activation. The quantizer can then no longer merge
the activation (ReLU6 or ReLU) into the convolution's output range, and that
tensor is quantized over its full pre-activation range: for MobileNetV2 and
EfficientNet-Lite0 about [-43, 60] instead of [0, 6], a 17x coarser step
(ResNet-18: 2x). Mean padding keeps the convolution feeding its activation
directly, so the folded INT8 models get the same first-convolution output
scale as the external ones (0.0235 for MobileNetV2 and EfficientNet-Lite0,
0.0191 for ResNet-18). Measured on ImageNetV2 eval-quick (INT8, CPU EP, not the
published evaluation):

| Model | INT8 folded top-1, correction after the conv | INT8 folded top-1, mean padding | INT8 external top-1 |
|---|---|---|---|
| MobileNetV2 | 56.8% | 59.0% | 58.9% |
| ResNet-18 | 57.5% | 57.4% | 57.3% |
| EfficientNet-Lite0 | 60.1% | 62.0% | 61.6% |

Measured on 64 ImageNetV2 images (FP32, ONNX Runtime CPU EP), folded file on
[0, 1] input vs external file on normalized input, all batch sizes:

| Model | Mean-padding fold: max abs diff, top-1 agreement | Naive fold, zero padding kept (not shipped) |
|---|---|---|
| MobileNetV2 | 6.9e-6 to 8.1e-6, 100% | 0.60, 96.9% |
| ResNet-18 | 1.7e-5, 100% | 2.31, 93.8% |
| EfficientNet-Lite0 | 8.6e-6 to 9.5e-6, 100% | 1.04, 95.3% |

The fold is at float rounding level (within the 1e-4 tolerance used for all
FP32 checks); the naive fold changes the top-1 class on 3-6% of images.

### INT8 folded models

The folded FP32 files are quantized with the same recipe as the others
([INT8 QDQ models](#int8-qdq-models)), except that the calibration images are
not normalized. Batch sizes share scales and zero points, pruning removes the
same dead chains, and the unpruned copies are kept. Metadata keeps the folded
input contract (`edgebench.folded_mean`, `edgebench.folded_std`,
`edgebench.input_range`, `edgebench.fold_padding = mean_pad`) and points
`edgebench.source_model` at the folded FP32 file. The quantizer stores the
mean strips as INT8 constants with their own scale.

Sanity numbers on ImageNetV2 eval-quick (1,000 images; **not the published
evaluation**), ONNX Runtime CPU EP:

| Model | INT8 external top-1 | INT8 folded top-1 | INT8 folded vs INT8 external | INT8 folded vs FP32 folded |
|---|---|---|---|---|
| MobileNetV2 | 58.9% | 59.0% | 93.7% | 92.2% |
| ResNet-18 | 57.3% | 57.4% | 95.2% | 95.1% |
| EfficientNet-Lite0 | 61.6% | 62.0% | 96.8% | 95.4% |

The folded INT8 models are as faithful to FP32 as the external ones (INT8 vs
FP32 agreement 92.6%, 94.6% and 95.2% for the external files).

On the NPU (QNN HTP, strict mode), every folded INT8 batch-1 model creates a
session and runs; NPU and CPU EP agree on the top-1 class for 62 of 64
ImageNetV2 images for all three models. Node counts are the external INT8
model's plus ten (the two `Concat` nodes, a DequantizeLinear per strip and a
QuantizeLinear/DequantizeLinear pair after each `Concat`): MobileNetV2 313
(303), ResNet-18 150 (140), EfficientNet-Lite0 295 (285).

### Making and verifying them

```powershell
.\.venv\Scripts\python.exe scripts\fold_models.py
.\.venv\Scripts\python.exe scripts\quantize_models.py --normalization folded

.\.venv\Scripts\python.exe scripts\fold_models.py --verify
.\.venv\Scripts\python.exe scripts\quantize_models.py --verify   # external and folded
```

`fold_models.py` refuses an FP32 file that does not match `models/manifest.json`,
checks every folded file against the external one on 64 images (1e-4, top-1
identical) and batch 4 and 8 against batch 1, and is deterministic: folding
again gives byte-identical files.

## Recipes and the manifest

Three things describe each model file, with one source of truth:

| What | Where | Committed | Role |
|---|---|---|---|
| ONNX metadata (`edgebench.*`) | inside the `.onnx` file | with the file | the source: written by export, fold and quantize (and `prepare_autoencoder.py`) |
| Recipe sidecar | `<name>.recipe.json` next to the file | no (travels with the file) | a structured, readable copy of the metadata |
| Manifest | `models/manifest.json` | yes | the index: every file, its hash and what it was made from |

### Recipe sidecar

`<name>.recipe.json` is built from the model file alone: its metadata, its
graph inputs and outputs, its SHA256 and size (`scripts/common/model_recipe.py`).
So it can be regenerated from the `.onnx` file at any time, and a file without
a sidecar loses nothing. Schema: `schemas/model-recipe.v1.1.0.schema.json`
(recipe version `1.1.0`; the `1.0.0` schema is kept for older sidecars). Groups:

| Group | Fields |
|---|---|
| `identity` | file, model, variant, precision, normalization, batch size, SHA256, size |
| `weights` | library (`torchvision`/`timm`/`colab_bundle`), weights id, weights-file SHA256; URL for downloaded weights, and for timm also the Hugging Face repository, revision and file; for an imported model the committed source file's path, the notebook and its commit, the commit it cloned, and the training record's SHA256 |
| `export` | exporter, opset, IR version, input and output names, dtypes and shapes, the versions of the tools that made the graph (torch/torchvision/onnx and timm, or tensorflow/keras/tf2onnx/onnx), and for imported models the local onnx version (`local_tool_versions`, `null` otherwise) |
| `preprocessing` | `kind` `image`: resize, interpolation, crop, scale, layout, dtype, mean/std, where normalization is applied (by the caller, or folded into the model), what the input tensor holds, and for folded files the padding method and first convolution. `kind` `time_series`: dataset, split manifest and its SHA256, window length and stride, layout, dtype, mean/std and the split they come from, where normalization is applied, the input range, and why there is no folded variant |
| `graph_cleanup` | imported models: what the exact clean-up rewrote (nodes before and after, bias Adds folded, Squeeze/Unsqueeze pairs removed, reshape chains collapsed), on the cleaned file and every file made from it; `null` otherwise |
| `quantization` | INT8 only (`null` for FP32): format, method, weight type and symmetry, per-channel, activation type and symmetry, bias type, calibration (method, dataset, calibration-manifest SHA256, sample count, batch), QNN preprocessing, pruned or not, chains and nodes pruned, quantizer and ONNX Runtime versions |
| `lineage` | the parent file and its SHA256 (`null` for a root: an FP32 export, made from the weights, or an imported source graph) |

A key the recipe needs that is missing from the metadata is an error that names
the key; no field is filled with a guess or `null`. The schema allows `null`
only where it means "not applicable": `quantization` for FP32 files,
`preprocessing.fold` for external normalization, `graph_cleanup` and
`export.local_tool_versions` for exported models, and `lineage.parent` for
roots. Image files carry no `kind` in their `edgebench.input_preprocessing`
(they predate it); their kind is `image`. For an imported model,
`model_recipes.py --check` also compares the weights block with the registry
pins.

### Manifest

Each entry of `models/manifest.json` (manifest version `1.1.0`, schema
`schemas/model-manifest.v1.1.0.schema.json`; 1.1.0 adds the variant
`fp32_imported`) records `file`, `sha256`, `size_bytes`, `model`, `variant`,
`precision`, `normalization`, `batch_size`, `pruned` (INT8 only, `null` for
FP32) and `parent_sha256` (`null` for roots). Every field except the hash and
size comes from the file's metadata. The parents form a tree per model and
batch size:

```
<model>_fp32_b<n>                      (exported from the weights)
  <- <model>_int8_b<n>, <model>_int8_unpruned_b<n>
  <- <model>_fp32_folded_b<n>
       <- <model>_int8_folded_b<n>, <model>_int8_folded_unpruned_b<n>

conv1d_autoencoder_fp32_imported_b<n>  (imported: the pinned source file + metadata)
  <- conv1d_autoencoder_fp32_b<n>      (cleaned graph)
       <- conv1d_autoencoder_int8_b<n>, conv1d_autoencoder_int8_unpruned_b<n>
```

Lineage check: every parent SHA256 is in the manifest; following parents
always ends at a root with FP32 precision and external normalization of the
same model and batch size: the FP32 export of an exported model, the
`fp32_imported` file of a model in `IMPORTED_MODELS` (an imported model's
`fp32` file must have that parent, an exported model may have no
`fp32_imported` file); each step goes to the expected parent variant; and there
are no cycles or duplicate hashes (`lineage_problems` in
`scripts/common/model_registry.py`).

### Regenerating and checking

Export, fold and quantize write the recipe sidecar and the manifest entry of
every file they make. For files that already exist:

```powershell
# rewrite every sidecar from its model file
.\.venv\Scripts\python.exe scripts\model_recipes.py

# write nothing; fail if anything disagrees
.\.venv\Scripts\python.exe scripts\model_recipes.py --check
```

`--check` validates the manifest against its schema and its lineage, then for
every file on disk checks that the sidecar exists, matches the recipe schema
and equals the recipe rebuilt from the file, and that the manifest entry equals
what the file's metadata, hash and size say. Files listed but not on disk are
reported and skipped. The manifest, lineage and recipe-schema tests
(`scripts/tests/test_model_recipes.py`) need only the standard library and
jsonschema and run in CI; the tests on real files skip when the files are
absent.

### Inspecting a model file

```powershell
.\.venv\Scripts\python.exe scripts\model_info.py models\efficientnet_lite0\efficientnet_lite0_int8_folded_b4.onnx
```

prints the recipe (from the sidecar when it matches the file's SHA256,
otherwise from the file's metadata) and the lineage, followed through the
manifest:

```
lineage
     efficientnet_lite0/efficientnet_lite0_int8_folded_b4.onnx (int8_folded, sha256 1b28616d545a)
  <- efficientnet_lite0/efficientnet_lite0_fp32_folded_b4.onnx (fp32_folded, sha256 dd64f81b21ea)
  <- efficientnet_lite0/efficientnet_lite0_fp32_b4.onnx (fp32, sha256 0ec46a6818d0)
```

`--json` prints the recipe document instead.

## Reference outputs

`models/references/` holds an answer key for every file in the manifest: what
Python's ONNX Runtime computes for that file on two fixed input sets. The C++
engine is tested against it, so a benchmark number always comes from an engine
that is known to compute the same outputs as the reference implementation.

### Input sets

| Input set | Contents | Runs per file |
|---|---|---|
| `windows-reference-<hash>` (the autoencoder) | the 8 reference windows of the split manifest (one centred in each labelled anomaly of the test split, 4 normal ones), normalized as the manifest says; `<hash>` is the start of the split manifest's SHA256 | batch 1: 8, batch 4: 2, batch 8: 1 |
| `images-normalized-<hash>` / `images-unit` | 8 ImageNetV2 eval-quick images, preprocessed with `scripts/common/preprocess.py` (resize 256, crop 224, scale to [0, 1]); normalized with the model's own mean/std for external-normalization files (`<hash>` identifies the mean/std), left in [0, 1] for folded files, as each file's recipe says | batch 1: 8, batch 4: 2, batch 8: 1 |
| `splitmix64-seed0-<shape>` (e.g. `splitmix64-seed0-1x3x224x224`) | the input `edgebench run` generates and times by default (`--input random --seed 0`), reproduced in Python from the documented generator ([cli.md](cli.md#input-data)) | 1 |

The images are a seeded sample of distinct classes:
`random.Random(20261124).sample` over eval-quick in (label, path) order, the
first 8 distinct classes (labels 843, 177, 691, 975, 3, 45, 688, 815).
`models/references/inputs.json` records the rule, each image's path, label
and SHA256, and the SHA256 of every input tensor.

### What is stored

| Device | Files | Session |
|---|---|---|
| `cpu` | all 66 | CPU EP, 4 intra-op threads, sequential execution, all graph optimizations, `session.intra_op.allow_spinning` 1 (the engine's CPU provider settings) |
| `npu` | the 42 INT8 files | QNN HTP, strict (`session.disable_cpu_ep_fallback` 1, so session creation fails if any node would run on the CPU), burst performance mode |
| `npu_fp16` | the 24 FP32 files | the same, with `enable_htp_fp16_precision` 1: the NPU computes FP32 models in FP16 (`edgebench run --npu-fp16`) |

Tensors are raw little-endian float32 files (the format
`tools/ort_probe/make_reference.py` already used), listed in
`models/references/index.tsv` with input set, model file, device, run, the
input files of the run, shape and SHA256. One JSON record per file and device
(`outputs/<model>/<file stem>/<device>.json`) holds the model file's SHA256,
the input sets, device and execution provider, thread count or provider
options, the ONNX Runtime version (and for the NPU the QNN execution provider
version and the QNN backend version from `QnnHtp.dll`), and every output's
SHA256 and top-1 classes (for the autoencoder: each window's reconstruction
MSE). NPU records also say whether running the same input twice gave identical
outputs, and the top-1 agreement with the CPU on the images (for the
autoencoder: the largest difference from the CPU on the windows). Like the
models, the references are not committed (24 MB).

```powershell
# write all references (CPU and NPU; --no-npu for CPU only); about 3 minutes
.\.venv\Scripts\python.exe scripts\make_references.py

# write nothing; fail on missing, stale or damaged references
.\.venv\Scripts\python.exe scripts\make_references.py --check
```

A reference is stale when the model SHA256 in its record is not the current
file's: regenerating a model file (for example after a metadata change)
requires regenerating the references. `--check` also verifies every tensor
against the SHA256 in the index and that the image selection and the reference
windows are still the documented ones.

### What the C++ tests prove

`engine/tests/reference_models_test.cpp` (suite `ReferenceModels`, ctest label
`models`) has one test per model file. Each creates the session exactly as
`edgebench run` does, through the CPU provider with IoBinding and
preallocated buffers (the timed mode), 4 threads, and checks:

- the input the engine generated for `--input random --seed 0` equals the
  Python reproduction byte for byte;
- every output equals the Python output byte for byte, on the generated input
  and on the images (the reference windows for the autoencoder).

No tolerance: the engine and the Python package use the same ONNX Runtime
version, thread count and session settings, and all 66 files (FP32 and INT8,
external and folded, pruned and unpruned, imported and cleaned, all batch
sizes) match bit for bit.
A further test checks that the index lists exactly the files the tests cover.
Without the model files or the references the tests are skipped with the
reason (CI has neither).

On the NPU (suite `ReferenceModelsNpu`, label `npu`) the engine's outputs are
bit-identical to the Python ONNX Runtime references on the same device and
versions, for every INT8 file (`npu`) and every FP32 file run with FP16 compute
(`npu_fp16`), at every batch size, on both input sets, through the QNN provider
in strict mode with IoBinding; `ReferenceDeterminismNpu` adds that fresh
sessions and both run bindings give identical outputs. NPU and CPU outputs of
the same INT8 file differ (different kernels and accumulation), with top-1
agreement on the 8 images of 8/8 for every INT8 file except EfficientNet-Lite0's
folded ones (7/8).

### Batch equivalence

Whether a file at batch 4 or 8 computes, for every item, exactly the output its
batch-1 file computes for the same item follows from the references above: the
8 images (or windows) are the same at every batch size, so item *i* of a batch-4
or batch-8 run is compared byte for byte with run *i* of the batch-1 file, on
the same device. The SplitMix64 input does not count, because its batch items
differ from each other. `scripts/make_references.py` writes the result to
`models/batch_equivalence.json` (committed, about 80 KB) whenever it writes the
references, and `--check` fails if the committed file no longer matches them:

```powershell
# rewrite only the summary, from the references on disk
.\.venv\Scripts\python.exe scripts\make_references.py --batch-equivalence
```

Each entry is keyed by the model file's SHA256, the provider (`cpu`, `npu`)
and the compute precision (the stored precision on the CPU; `int8` or `fp16`
on the NPU), and records `equivalent_to_b1`, the items compared and differing,
and the largest absolute difference. `edgebench run` looks every model up there
and records the answer in its result (`execution.batch_equivalence`; a model
that is not equivalent gets a `batch_not_equivalent` warning, see
[cli.md](cli.md#batch-equivalence)). The summary lists 132 entries (every model
file on each of its reference devices), 22 of them not equivalent. By group
("same": every file of the group computes the batch-1 outputs exactly;
"differs": none does):

| Model | Variants | Provider, compute | b4 vs b1 | b8 vs b1 | Largest difference |
|---|---|---|---|---|---|
| `mobilenetv2` | FP32, FP32 folded | CPU, FP32 | differs | differs | 3.8e-06 |
| `mobilenetv2` | FP32, FP32 folded | NPU, FP16 | same | same | 0 |
| `resnet18` | FP32, FP32 folded | CPU, FP32 | same | same | 0 |
| `resnet18` | FP32, FP32 folded | NPU, FP16 | same | same | 0 |
| `efficientnet_lite0` | FP32, FP32 folded | CPU, FP32 | differs | differs | 5.2e-06 |
| `efficientnet_lite0` | FP32, FP32 folded | NPU, FP16 | same | same | 0 |
| `mobilenetv2` | INT8 (all four) | CPU, INT8 | same | same | 0 |
| `mobilenetv2` | INT8 (all four) | NPU, INT8 | same | differs | 0.059 |
| `resnet18` | INT8 (all four) | CPU, INT8 | same | same | 0 |
| `resnet18` | INT8 (all four) | NPU, INT8 | same | differs | 0.17 |
| `efficientnet_lite0` | INT8 (all four) | CPU, INT8 | same | same | 0 |
| `efficientnet_lite0` | INT8 (all four) | NPU, INT8 | same | differs | 0.085 |
| `conv1d_autoencoder` | FP32 imported | CPU, FP32 | same | same | 0 |
| `conv1d_autoencoder` | FP32 imported | NPU, FP16 | differs | differs | 0.0078 |
| `conv1d_autoencoder` | FP32 | CPU, FP32 | same | same | 0 |
| `conv1d_autoencoder` | FP32 | NPU, FP16 | same | same | 0 |
| `conv1d_autoencoder` | INT8, INT8 unpruned | CPU, INT8 | same | same | 0 |
| `conv1d_autoencoder` | INT8, INT8 unpruned | NPU, INT8 | same | same | 0 |

The CPU differences of the FP32 MobileNetV2 and EfficientNet-Lite0 files are in
the last bits of the logits (largest 3.8e-06 and 5.2e-06; every item differs).
Both models use depthwise convolutions and ResNet-18 does not, so a different
accumulation order in ONNX Runtime's CPU kernels for a batch is the likely
cause (not verified). ResNet-18 and every INT8 file are exact on the CPU. On the NPU, the INT8 image files at batch 8 compute
different outputs on every image (the HTP runs the batch-8 graph differently;
batch 4 is exact), and the raw tf2onnx graph of the autoencoder
(`fp32_imported`) differs at batch 4 and 8 in FP16, while its cleaned graph does
not. A difference is a fact about the computation, not an error: the
published accuracy evaluation uses the largest batch whose top-5 predictions
match batch 1 ([accuracy.md](accuracy.md)), and latency results at a batch size
that is not equivalent say so.
