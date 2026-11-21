# ml/quantization

Quantization of FP32 ONNX models (e.g. QDQ INT8 for the Qualcomm NPU), including
calibration data handling.

The MobileNetV2 INT8 QDQ quantization is `scripts/quantize_qnn.py`; calibration
data comes from `scripts/get_data.py`.
