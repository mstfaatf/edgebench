# models

Model files used by benchmarks. Binaries (`*.onnx` etc.) are **not committed**.
They are produced locally by the pipelines in `ml/` (export, then optional
quantization) or, for the Keras model, exported in Colab and copied here.
Small test models live under `engine/tests/fixtures/` instead.
