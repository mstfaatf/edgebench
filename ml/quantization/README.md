# ml/quantization

Quantization of FP32 ONNX models (e.g. QDQ INT8 for the Qualcomm NPU), including
calibration data handling.

Static INT8 QDQ quantization of every exported model is
`scripts/quantize_models.py` (recipe in [docs/models.md](../../docs/models.md#int8-qdq-models));
calibration data comes from `scripts/get_data.py`. The feasibility study's
MobileNetV2 variants come from `scripts/quantize_qnn.py`.
