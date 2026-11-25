# models

Model files used by benchmarks. Binaries (`*.onnx` etc.) are **not committed**.
They are produced locally by the export and quantization scripts in
`scripts/`; the Keras autoencoder's source files are the exception (below).
Small models needed by tests belong in a `tests/fixtures/` directory next to
the tests that use them (with `sources/`, the only places `.gitignore` allows
model files).

FP32 models are produced by `scripts/export_models.py`: one directory per
model (`mobilenetv2/`, `resnet18/`, `efficientnet_lite0/`), one file per batch size
(`<model>_fp32_b<batch>.onnx`). `manifest.json` (committed) lists every model
file with its SHA256, size, what it is and the SHA256 of the file it was made
from; `export_models.py --verify` checks a fresh export against it.
`scripts/fold_models.py` writes normalization-folded FP32 variants
(`<model>_fp32_folded_b<batch>.onnx`, input in [0, 1]), and
`scripts/quantize_models.py` writes the INT8 QDQ versions of both next to them
(`<model>_int8_b<batch>.onnx`, plus unpruned references
`<model>_int8_unpruned_b<batch>.onnx`). The feasibility study's MobileNetV2
INT8 variants (`mobilenetv2_int8qdq_*`, `unpruned/`) come from
`scripts/quantize_qnn.py`. See [docs/models.md](../docs/models.md) and
[docs/setup.md](../docs/setup.md#preparing-data-and-models). Every model has a
JSON sidecar with the same name recording the checks run on it, and a
`<name>.recipe.json` sidecar describing how it was made, generated from the
file's own metadata (`scripts/model_recipes.py` regenerates and checks them,
`scripts/model_info.py <file.onnx>` prints one).

The Conv1D time-series autoencoder is trained in Colab, not exported here. Its
exported FP32 files, training record and reference arrays are committed in
`sources/conv1d_autoencoder/` (pinned by SHA256; the only committed model files
besides test fixtures), and `scripts/prepare_autoencoder.py` makes
`conv1d_autoencoder/` from them: `<model>_fp32_imported_b<batch>.onnx` (the
source graph with metadata), `<model>_fp32_b<batch>.onnx` (the cleaned graph)
and the INT8 files. See [docs/autoencoder.md](../docs/autoencoder.md).

Qualcomm AI Hub's MobileNetV2 (w8a8 and float ONNX, used as a baseline) is
downloaded to `aihub/` by `scripts/get_aihub_baseline.py`.
