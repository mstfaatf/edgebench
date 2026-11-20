# models

Model files used by benchmarks. Binaries (`*.onnx` etc.) are **not committed**.
They are produced locally by the pipelines in `ml/` (export, then optional
quantization) or, for the Keras model, exported in Colab and copied here.
Small models needed by tests belong in a `tests/fixtures/` directory next to
the tests that use them (the only place `.gitignore` allows model files).
