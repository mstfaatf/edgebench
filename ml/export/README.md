# ml/export

Scripts that export trained models to ONNX with fixed input shapes and a pinned
opset, and check the exported graph with `onnx.checker`.

The torchvision FP32 export is `scripts/export_models.py` (models listed in
`scripts/common/model_registry.py`); see [docs/models.md](../../docs/models.md).
