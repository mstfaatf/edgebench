# ml/validation

Accuracy checks comparing ONNX Runtime outputs (CPU and NPU) against the
framework reference outputs, so a benchmarked model is known to produce correct
results.

For MobileNetV2 the checks are scripts: `scripts/eval_cpu.py` (CPU EP),
`scripts/npu_check.py` (NPU, strict mode, agreement with the CPU),
`scripts/compare_aihub.py` (Qualcomm AI Hub baseline) and
`tools/ort_probe/make_reference.py` (reference tensors for C++ vs Python);
results are summarized in [docs/feasibility.md](../../docs/feasibility.md).
