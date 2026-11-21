# models

Model files used by benchmarks. Binaries (`*.onnx` etc.) are **not committed**.
They are produced locally by the pipelines in `ml/` (export, then optional
quantization) or, for the Keras model, exported in Colab and copied here.
Small models needed by tests belong in a `tests/fixtures/` directory next to
the tests that use them (the only place `.gitignore` allows model files).

MobileNetV2 lives in `mobilenetv2/` and is produced by
`scripts/export_mobilenetv2.py` (FP32) and `scripts/quantize_qnn.py`
(INT8 QDQ); see [docs/setup.md](../docs/setup.md#preparing-data-and-models).
Every model has a JSON sidecar with the same name recording how it was made
and its SHA256.

Qualcomm AI Hub's MobileNetV2 (w8a8 and float ONNX, used as a baseline) is
downloaded to `aihub/` by `scripts/get_aihub_baseline.py`.
