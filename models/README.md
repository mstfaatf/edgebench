# models

Model files used by benchmarks. Binaries (`*.onnx` etc.) are **not committed**.
They are produced locally by the pipelines in `ml/` (export, then optional
quantization) or, for the Keras model, exported in Colab and copied here.
Small models needed by tests belong in a `tests/fixtures/` directory next to
the tests that use them (the only place `.gitignore` allows model files).

FP32 models are produced by `scripts/export_models.py`: one directory per
model (`mobilenetv2/`, `resnet18/`, `efficientnet_lite0/`), one file per batch size
(`<model>_fp32_b<batch>.onnx`). `manifest.json` (committed) lists the SHA256
and size of every exported file; `export_models.py --verify` checks a fresh
export against it. `scripts/fold_models.py` writes normalization-folded FP32 variants
(`<model>_fp32_folded_b<batch>.onnx`, input in [0, 1]), and
`scripts/quantize_models.py` writes the INT8 QDQ versions of both next to them
(`<model>_int8_b<batch>.onnx`, plus unpruned references
`<model>_int8_unpruned_b<batch>.onnx`). The feasibility study's MobileNetV2
INT8 variants (`mobilenetv2_int8qdq_*`, `unpruned/`) come from
`scripts/quantize_qnn.py`. See [docs/models.md](../docs/models.md) and
[docs/setup.md](../docs/setup.md#preparing-data-and-models). Every model has a
JSON sidecar with the same name recording how it was made and its SHA256.

Qualcomm AI Hub's MobileNetV2 (w8a8 and float ONNX, used as a baseline) is
downloaded to `aihub/` by `scripts/get_aihub_baseline.py`.
