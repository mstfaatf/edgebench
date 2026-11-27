"""Run the CPU/NPU comparison matrix through `edgebench run` (local results).

The configurations are those of ``common/matrix.py``. Everything goes into one
new directory ``<output-root>/<UTC stamp>/``:

    doctor.json        `edgebench doctor --json`, run first
    plan.json          settings, every run's arguments, exit code and result file
    qnn-context/       the EP context cache of this matrix (--ep-cache-dir)
    cache_build/       one `--ep-cache build` result per NPU run (built just before it)
    results/           the timed results (and their latencies)
    logs/<run>.log     stdout and stderr of every edgebench call

Each NPU run's cache entry is built right before the run, so the timed run must
find it (a cache hit); building just before matters because a build deletes the
entries of the same model file with another key (e.g. another
--htp-graph-finalization-mode). The run stops before anything is timed if
doctor fails, or if its power or background_load check is not a pass (unless
--allow-doctor-warnings). Check the results with ``scripts/check_matrix.py``.

The numbers are local measurements for checking the engine, not published
results.

Usage:
    python scripts/run_matrix.py [--trials 3] [--iterations 100] [--warmup 20]
                                 [--cooldown-ms 1000] [--only RUN_ID ...]
                                 [--exe PATH] [--output-root DIR]
                                 [--allow-doctor-warnings]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import matrix as mx  # noqa: E402
from common.model_registry import MODELS_DIR, REPO_ROOT  # noqa: E402

DEFAULT_EXE = REPO_ROOT / "build" / "arm64-release" / "bin" / "Release" / "edgebench.exe"
OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "local" / "matrix"
DOCTOR_GATES = ("power", "background_load")


def call(exe: Path, args: list[str], log: Path) -> tuple[int, float, str]:
    start = time.perf_counter()
    p = subprocess.run([str(exe), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    seconds = time.perf_counter() - start
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"$ edgebench {' '.join(args)}\n--- exit {p.returncode} in {seconds:.1f} s\n"
                f"--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n")
    return p.returncode, seconds, p.stdout


def result_file(directory: Path, name: str) -> str | None:
    files = [f for f in directory.glob(f"{name}_*.json") if not f.name.endswith("_latest.json")]
    return files[0].relative_to(directory.parent).as_posix() if len(files) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    runs = mx.planned_runs()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cooldown-ms", type=int, default=1000)
    parser.add_argument("--only", nargs="+", choices=[r.run_id for r in runs], metavar="RUN_ID")
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-doctor-warnings", action="store_true",
                        help="run even if doctor's power or background_load check is not a pass")
    args = parser.parse_args(argv)

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not args.exe.is_file():
        print(f"{args.exe} not found; build the engine first (docs/setup.md)", file=sys.stderr)
        return 2
    from common.results_io import run_stamp

    selected = [r for r in runs if args.only is None or r.run_id in args.only]
    missing = [r.file for r in selected if not (MODELS_DIR / r.file).is_file()]
    if missing:
        print(f"model files missing (run the model pipeline): {missing[:3]}", file=sys.stderr)
        return 1
    root = args.output_root / run_stamp()
    root.mkdir(parents=True)
    cache_dir = root / "qnn-context"
    logs = root / "logs"

    code, _, out = call(args.exe, ["doctor", "--json"], logs / "doctor.log")
    (root / "doctor.json").write_text(out, encoding="utf-8", newline="\n")
    try:
        doctor = json.loads(out)
        statuses = {c["name"]: c["status"] for c in doctor["checks"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"doctor printed no report (exit {code}); see {logs / 'doctor.log'}", file=sys.stderr)
        return 1
    print(f"doctor: exit {code}, " + ", ".join(f"{k} {v}" for k, v in statuses.items()))
    gates = {k: statuses.get(k) for k in DOCTOR_GATES}
    if code != 0 or (not args.allow_doctor_warnings and any(v != "pass" for v in gates.values())):
        print(f"stopping before any run: doctor exit {code}, "
              + ", ".join(f"{k} {v}" for k, v in gates.items())
              + " (published-grade runs need AC power, the Best performance power mode and an "
                "idle machine)", file=sys.stderr)
        return 1

    settings = {"trials": args.trials, "iterations": args.iterations, "warmup": args.warmup,
                "cooldown_ms": args.cooldown_ms}
    plan = {"matrix_version": mx.MATRIX_VERSION, "settings": settings,
            "exe": str(args.exe), "doctor": {"exit_code": code, "checks": statuses},
            "cache_dir": "qnn-context", "runs": []}
    started = time.perf_counter()
    failures = 0
    for i, run in enumerate(selected, 1):
        model = str(MODELS_DIR / run.file)
        entry = {"run_id": run.run_id, "model": run.model, "config": run.config,
                 "file": run.file, "batch": run.batch, "provider": run.provider,
                 "stored_precision": run.stored_precision,
                 "compute_precision": run.compute_precision,
                 "ep_compile_options": dict(run.ep_compile_options)}
        if run.npu:
            build = mx.cache_build_arguments(run, str(cache_dir))
            build_args = ["run", "--model", model, *build, "--output-dir",
                          str(root / "cache_build"), "--name", run.run_id]
            b_code, b_s, _ = call(args.exe, build_args, logs / f"{run.run_id}.log")
            entry["cache_build"] = {"args": build[:], "exit_code": b_code,
                                    "seconds": round(b_s, 1),
                                    "result": result_file(root / "cache_build", run.run_id)}
        timed = mx.run_arguments(run, cache_dir=str(cache_dir), **settings)
        r_code, r_s, _ = call(args.exe, ["run", "--model", model, *timed, "--output-dir",
                                         str(root / "results"), "--name", run.run_id],
                              logs / f"{run.run_id}.log")
        entry.update(args=timed, exit_code=r_code, seconds=round(r_s, 1),
                     result=result_file(root / "results", run.run_id))
        failures += r_code != 0 or (run.npu and entry["cache_build"]["exit_code"] != 0)
        plan["runs"].append(entry)
        print(f"[{i}/{len(selected)}] {run.run_id}: exit {r_code} in {r_s:.0f} s"
              + (f" (cache build exit {entry['cache_build']['exit_code']})" if run.npu else ""))
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8",
                                        newline="\n")
    plan["seconds"] = round(time.perf_counter() - started, 1)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8",
                                    newline="\n")
    print(f"{len(selected)} runs in {plan['seconds'] / 60:.1f} min, {failures} failed; "
          f"results in {root}")
    print(f"check them: python scripts/check_matrix.py {root}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
