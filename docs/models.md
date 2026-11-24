# Models

EdgeBench benchmarks ImageNet classifiers exported from PyTorch (torchvision and
timm) to ONNX with fixed shapes, one file per batch size, in FP32 and as static
INT8 QDQ models. This page describes which models, where their weights come
from, how they are exported, quantized and checked, and what input they expect.

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

Every exported file carries these `metadata_props`:

| Key | Example |
|---|---|
| `edgebench.model` | `mobilenetv2` |
| `edgebench.variant` | `fp32` |
| `edgebench.batch_size` | `4` |
| `edgebench.normalization` | `external` |
| `edgebench.input_preprocessing` | JSON: resize, crop, scale, mean, std, layout, dtype |
| `edgebench.weights` | `torchvision MobileNet_V2_Weights.IMAGENET1K_V2`, or `timm efficientnet_lite0.ra_in1k (timm/efficientnet_lite0.ra_in1k@<revision>)` |
| `edgebench.weights_sha256` | SHA256 of the weights file |
| `edgebench.exporter`, `edgebench.opset`, `edgebench.ir_version` | `torch.onnx.export (TorchScript, dynamo=False)`, `17`, `8` |
| `edgebench.torch_version`, `edgebench.torchvision_version`, `edgebench.onnx_version` | `2.13.0+cpu`, `0.28.0+8fb8771`, `1.23.2` |
| `edgebench.timm_version` (timm models only) | `1.0.31` |

## Files

```
models/
  manifest.json                       committed: SHA256 and size of every file below
  mobilenetv2/
    mobilenetv2_fp32_b1.onnx          + mobilenetv2_fp32_b1.json (sidecar)
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
```

Files are named `<model>_<variant>_b<batch>.onnx`, with variant `fp32`, `int8`
or `int8_unpruned`. The `.onnx` files and
sidecars are not committed (`models/` is gitignored except its README and
`manifest.json`). Each sidecar records the weights, exporter, versions, shapes,
ONNX metadata and the results of the checks below. The INT8 models of the
feasibility study (`scripts/quantize_qnn.py`) also live in `models/mobilenetv2/`.

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
its input (model, batch size, weights, `normalization`, `input_preprocessing`,
export versions), and rewrite or add:

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

The `edgebench.quant.*` keys are flat strings so a recipe file can be generated
from them. `edgebench run` reports these models as `stored_precision: int8`,
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
