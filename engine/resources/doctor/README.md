# Doctor probe models

`edgebench doctor` runs these models once on the NPU, in strict mode, with the
generated input of `edgebench run` (SplitMix64, seed 0), and compares the output
bytes with the expected outputs here. CMake builds all four files into the
executable (`cmake/EmbedFiles.cmake`), so doctor needs nothing besides the bundle.

| File | What |
|---|---|
| `npu_probe_int8.onnx` | `x` float32 [1, 3, 8, 8] -> 3x3 convolution (4 filters) -> Relu -> `y` [1, 4, 8, 8], static INT8 QDQ (uint8 activations, symmetric int8 weights, int32 bias) |
| `npu_probe_fp32.onnx` | The same convolution in FP32; doctor runs it with FP16 compute (`enable_htp_fp16_precision` = 1) |
| `npu_probe_int8.npu.f32` | Expected output of the INT8 model on the NPU (raw little-endian float32) |
| `npu_probe_fp32.npu_fp16.f32` | Expected output of the FP32 model on the NPU with FP16 compute |
| `probes.json` | Model and output SHA256, shapes, provider options and the versions the outputs were made with |

The models come from `scripts/make_test_models.py` (deterministic, needs only
`onnx`); the expected outputs from `scripts/make_probe_references.py`, which
runs the models with the Python reference sessions on the NPU (the C++ engine
matches those bit for bit). `make_probe_references.py --check` compares them
with fresh NPU outputs.
