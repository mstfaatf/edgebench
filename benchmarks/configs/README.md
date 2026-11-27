# benchmarks/configs

Suite configurations for `edgebench suite` (docs/suite.md): which models and
configurations run (an explicit list or a matrix), their run parameters,
repetitions, rotation, cooldowns and the pre-flight gate. Each file is
validated against `schemas/suite-config.v1.0.0.schema.json`
(`scripts/validate_suite.py`) and by the engine (`edgebench suite FILE --dry-run`).

| File | Purpose |
|---|---|
| `smoke.json` | A quick end-to-end check: the autoencoder on the CPU (FP32, INT8) and the NPU (INT8), batch 1, two repetitions, short runs. Not a benchmark |
| `autoencoder_npu_int8_b1.json` | The autoencoder, INT8 on the NPU, batch 1, three repetitions with the matrix's run parameters (cached EP context, init breakdown, profiled pass) |
