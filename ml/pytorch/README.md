# ml/pytorch

PyTorch model definitions and preparation code: loading pretrained models
(torchvision, timm), any fine-tuning, and producing the reference outputs used
to validate exported models. Runs natively on Windows ARM64 (see docs/setup.md).

`scripts/export_models.py` loads the torchvision weights (verified by SHA256)
and checks the exported ONNX models against PyTorch; see
[docs/models.md](../../docs/models.md).
