# Models

EdgeBench benchmarks ImageNet classifiers exported from torchvision to ONNX
with fixed shapes, one file per batch size. This page describes which models,
where their weights come from, how they are exported and checked, and what
input they expect.

## Supported models

| Model | Name | torchvision weights | Weights file (SHA256) | ImageNet-1k top-1 (torchvision) |
|---|---|---|---|---|
| MobileNetV2 | `mobilenetv2` | `MobileNet_V2_Weights.IMAGENET1K_V2` | [`mobilenet_v2-7ebf99e0.pth`](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth) (`7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`, 14,258,573 bytes) | 72.154 |
| ResNet-18 | `resnet18` | `ResNet18_Weights.IMAGENET1K_V1` | [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) (`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, 46,830,571 bytes) | 69.758 |

- MobileNetV2 uses `IMAGENET1K_V2`, torchvision's improved training recipe.
- ResNet-18 uses `IMAGENET1K_V1`, the only ImageNet weights torchvision
  provides for it (the original training recipe), so the two models were not
  trained the same way.
- The weights are downloaded by torchvision into its cache
  (`%USERPROFILE%\.cache\torch\hub\checkpoints`). torchvision's file names and its
  download check only use the first 8 hex digits of the SHA256, so the export
  verifies the full SHA256 pinned in `scripts/common/model_registry.py` before
  using a file, and stops if it differs.
- torchvision's top-1 figures are on the ImageNet-1k validation set with
  torchvision's own evaluation transform (for MobileNetV2 `IMAGENET1K_V2` that
  resizes to 232, not 256). EdgeBench evaluates on ImageNetV2 with one shared
  transform, so its numbers are not comparable with these.

The model list is `MODELS` in `scripts/common/model_registry.py`. To add a
torchvision classifier, add a `ModelSpec` there (constructor, weights enum and
member, weights URL, SHA256 and size), run the export for it, and commit the
updated manifest.

## Export settings

| Setting | Value |
|---|---|
| Exporter | `torch.onnx.export`, TorchScript-based (`dynamo=False`) |
| Opset | 17 (default domain); IR version 8 |
| Precision | FP32 weights and activations |
| Input | `input`, float32 `[batch, 3, 224, 224]`, NCHW, RGB, normalized (see below) |
| Output | `logits`, float32 `[batch, 1000]`, raw logits (no softmax), torchvision class order |
| Batch sizes | 1, 4, 8: one file each, batch dimension fixed |
| Dynamic axes | none: every graph input and output has concrete dimensions |
| Graph | constant folding as done by the exporter; no further optimization (ONNX Runtime optimizes at session creation) |
| Versions used | torch 2.13.0+cpu, torchvision 0.28.0, onnx 1.23.2 (recorded in `models/manifest.json`) |

These are the settings the feasibility study validated on the CPU and QNN
execution providers ([feasibility.md](feasibility.md)); the MobileNetV2 batch-1
graph is identical to the one used there, with metadata added.

### Input: normalization is outside the model

The files expect input that is **already normalized**: the output of
`scripts/common/preprocess.py` `load_image()`, which resizes the shorter side
to 256 (bilinear), center-crops 224, scales to [0, 1] and applies the ImageNet
mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Feeding [0, 1]
or 0-255 pixels gives wrong predictions without any error. Preprocessing is
never part of the timed region (see [methodology.md](methodology.md)).

This is recorded in the file as ONNX metadata
`edgebench.normalization = external`. `edgebench run` reads that key and writes
it to the result's `model.normalization`. A later variant with the
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
| `edgebench.weights` | `torchvision MobileNet_V2_Weights.IMAGENET1K_V2` |
| `edgebench.weights_sha256` | SHA256 of the weights file |
| `edgebench.exporter`, `edgebench.opset`, `edgebench.ir_version` | `torch.onnx.export (TorchScript, dynamo=False)`, `17`, `8` |
| `edgebench.torch_version`, `edgebench.torchvision_version`, `edgebench.onnx_version` | `2.13.0+cpu`, `0.28.0+8fb8771`, `1.23.2` |

## Files

```
models/
  manifest.json                     committed: SHA256 and size of every exported file
  mobilenetv2/
    mobilenetv2_fp32_b1.onnx        + mobilenetv2_fp32_b1.json (sidecar)
    mobilenetv2_fp32_b4.onnx
    mobilenetv2_fp32_b8.onnx
  resnet18/
    resnet18_fp32_b1.onnx
    ...
```

Files are named `<model>_<variant>_b<batch>.onnx`. The `.onnx` files and
sidecars are not committed (`models/` is gitignored except its README and
`manifest.json`). Each sidecar records the weights, exporter, versions, shapes,
ONNX metadata and the results of the checks below. The INT8 models of the
feasibility study (`scripts/quantize_qnn.py`) also live in `models/mobilenetv2/`.

## Exporting and verifying

```powershell
# all models, batch sizes 1, 4 and 8
.\.venv\Scripts\python.exe scripts\export_models.py

# a subset
.\.venv\Scripts\python.exe scripts\export_models.py --model resnet18 --batch 1 8

# re-export in memory and compare with models/manifest.json (writes nothing)
.\.venv\Scripts\python.exe scripts\export_models.py --verify
```

`scripts/export_mobilenetv2.py` is kept as a shortcut for
`export_models.py --model mobilenetv2 --batch 1`.

Exporting is deterministic: exporting again gives byte-identical files, which
`--verify` checks against the committed manifest. That holds for the same
torch, torchvision and onnx versions; with other versions `--verify` says so
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
ImageNetV2 eval-quick images (64 different classes):

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
ResNet-18). The measured differences are 1e-5 to 2.3e-5, so 1e-4 leaves a 4-9x
margin. A real export error (wrong weights, layout or layer) changes logits by
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

As a sanity check of the export (not a published accuracy figure), top-1 /
top-5 on ImageNetV2 eval-quick (1,000 images, one per class) is 58.7% / 81.2%
for MobileNetV2 and 56.4% / 79.8% for ResNet-18. A broken export or label
mapping scores near 0.1%.

The tests in `scripts/tests/test_export_models.py` repeat these checks (and
skip with a reason when torch, the weights, the exported files or ImageNetV2
are missing); `OnnxInspectModels.ExportedFp32ModelsHaveFixedShapes` (CTest
label `models`) checks the shapes and metadata with the engine's own ONNX
reader.
