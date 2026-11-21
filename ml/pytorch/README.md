# ml/pytorch

PyTorch model definitions and preparation code: loading pretrained models
(torchvision, timm), any fine-tuning, and producing the reference outputs used
to validate exported models. Runs natively on Windows ARM64 (see docs/setup.md).

For MobileNetV2, `scripts/export_mobilenetv2.py` loads the torchvision weights
and checks the exported ONNX model against PyTorch.
