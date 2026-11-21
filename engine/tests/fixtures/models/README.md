# Test models

Tiny ONNX models for the C++ tests (a few hundred bytes each):

| File | Input | Output | Used to test |
|---|---|---|---|
| `fixed_float32.onnx` | `x` float32 [1, 4] | `y` float32 [1, 3] | Sessions, run modes, node placement |
| `dynamic_batch.onnx` | `x` float32 [batch, 4] | `y` float32 [batch, 3] | Rejection of dynamic dimensions |
| `int64_input.onnx` | `ids` int64 [1, 4] | `y` float32 [1, 4] | Rejection of non-float32 inputs |

`fixed_float32` and `dynamic_batch` compute `relu(x W + b)` with fixed constants;
`int64_input` casts its input to float32. Regenerate them (deterministically)
with:

```powershell
.\.venv\Scripts\python.exe scripts\make_test_models.py
```

The script only needs the `onnx` package.
