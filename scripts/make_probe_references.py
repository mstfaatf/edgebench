"""Write the expected NPU outputs of the `edgebench doctor` probe models.

`edgebench doctor` runs the two probe models of engine/resources/doctor/ (made by
scripts/make_test_models.py) once on the NPU in strict mode, with the engine's
generated input (SplitMix64, seed 0), and compares the output bytes with the
files this script writes next to them:

- npu_probe_int8.npu.f32: npu_probe_int8.onnx, INT8 compute;
- npu_probe_fp32.npu_fp16.f32: npu_probe_fp32.onnx, FP16 compute
  (enable_htp_fp16_precision = 1);
- probes.json: model and output SHA256, shapes, the provider options and the
  ONNX Runtime / QNN versions they were made with.

Raw little-endian float32, the format of the reference outputs in models/references/.
The sessions are the Python reference sessions (strict, burst; the C++ engine
matches them bit for bit). Each input is run twice and must give the same bytes.

    python scripts/make_probe_references.py           # write
    python scripts/make_probe_references.py --check   # compare, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.platform_guard import require_native_arm64  # noqa: E402

PROBE_DIR = Path(__file__).resolve().parent.parent / "engine" / "resources" / "doctor"
PROBES = {
    # model file -> (device, fp16)
    "npu_probe_int8.onnx": ("npu", False),
    "npu_probe_fp32.onnx": ("npu_fp16", True),
}
SEED = 0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_name(model: str, device: str) -> str:
    return f"{Path(model).stem}.{device}.f32"


def compute() -> tuple[dict[str, bytes], dict]:
    import numpy as np
    import onnxruntime as ort

    from common import onnx_quantize as oq
    from common import ort_qnn
    from common import reference_outputs as ro

    outputs: dict[str, bytes] = {}
    records = []
    for model, (device, fp16) in PROBES.items():
        path = PROBE_DIR / model
        session = oq.npu_session(path, fp16=fp16)
        inp = session.get_inputs()[0]
        x = ro.splitmix64_floats(SEED, int(np.prod(inp.shape))).reshape(inp.shape)
        runs = [session.run(None, {inp.name: x})[0] for _ in range(2)]
        if runs[0].tobytes() != runs[1].tobytes():
            raise SystemExit(f"{model}: two runs on the NPU gave different outputs")
        y = runs[0].astype("<f4", copy=False)
        name = output_name(model, device)
        outputs[name] = y.tobytes()
        provider_options = {"htp_performance_mode": "burst"}
        if fp16:
            provider_options["enable_htp_fp16_precision"] = "1"
        records.append({
            "model": model,
            "model_sha256": sha256(path.read_bytes()),
            "device": device,
            "compute_precision": "fp16" if fp16 else "int8",
            "input": {"kind": "splitmix64", "seed": SEED, "shape": list(inp.shape)},
            "output": {"file": name, "shape": list(y.shape), "dtype": "float32",
                       "sha256": sha256(outputs[name])},
            "provider_options": provider_options,
            "strict": True,
        })
    versions = {"onnxruntime": ort.__version__, **ort_qnn.qnn_versions()}
    return outputs, {"probes_version": "1.0.0", "versions": versions, "probes": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed files instead of writing them")
    args = parser.parse_args(argv)
    require_native_arm64()
    outputs, manifest = compute()
    if args.check:
        problems = [name for name, data in outputs.items()
                    if not (PROBE_DIR / name).is_file() or (PROBE_DIR / name).read_bytes() != data]
        for name in problems:
            print(f"differs: {name}")
        print("probe references: " + ("OK" if not problems else f"{len(problems)} differ"))
        return 1 if problems else 0
    for name, data in outputs.items():
        (PROBE_DIR / name).write_bytes(data)
        print(f"wrote {PROBE_DIR / name} ({len(data)} bytes)")
    text = json.dumps(manifest, indent=2) + "\n"
    (PROBE_DIR / "probes.json").write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {PROBE_DIR / 'probes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
