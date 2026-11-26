# QNN profiles

Profiling output of the QNN execution provider (EP options `profiling_level`
`basic` and `profiling_file_path`), written by real runs of
`edgebench run --provider npu --profile-accelerator --warmup 2
--profile-iterations 3` on `conv1d_autoencoder_int8_b1.onnx`. The files hold
timing events only (event name, time, unit). The C++ tests read them to check
the profile parser (`engine/tests/qnn_backend_info_test.cpp`).

| File | Session | Creation events |
|---|---|---|
| `compiled_basic.csv` | compiled from the ONNX model | graph finalization (`(finalize)`) |
| `context_basic.csv` | created from an EP context model | loading the context binary (`(load binary)`) |

Each holds 5 inferences (2 warm-up, 3 profiled).
