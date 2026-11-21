"""Shared machinery for the strict-mode NPU feasibility checks.

Used by ``scripts/npu_check.py`` and ``scripts/compare_aihub.py``. A check is a
list of :class:`RunConfig` (model + execution provider + I/O adapter); the
:class:`NpuHarness` then

- selects the QNN EP device whose hardware type is NPU,
- **places** each NPU config with a strict, verbose session (QNN validation
  log, ONNX Runtime node placement, one-inference profile), falling back to a
  non-strict diagnosis when the strict session cannot be created,
- measures **accuracy** on an Imagenette split with a separate quiet session,
- measures **timing** (cold init, EP context generation, cached init,
  steady-state latency), interleaving configurations with a cooldown and
  optionally repeating in several rounds with a rotated order,
- prints tables and assembles the environment part of the JSON summary.

All images go through ``common.preprocess`` once (to the [0, 1] RGB tensor);
an :class:`IOAdapter` turns that into what a particular model expects and turns
its raw output into float logits, so models with different I/O conventions see
exactly the same pixels.
"""

from __future__ import annotations

import gc
import json
import math
import platform
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn

from common import accuracy as acc
from common import ort_qnn
from common.preprocess import IMAGENETTE_ROOT, REPO_ROOT, load_image, normalize, read_split
from common.qnn_log import parse_qnn_profile_csv

HTP_PERFORMANCE_MODE = "burst"
DLL_FILTER = re.compile(r"onnxruntime|qnn", re.IGNORECASE)


# --------------------------------------------------------------------------- I/O adapters


@dataclass(frozen=True)
class IOAdapter:
    """Final tensor formatting for one model's input and output conventions."""

    name: str
    description: dict
    make_input: Callable[[np.ndarray], np.ndarray]  # [1,3,224,224] float32 in [0,1] -> model input
    to_logits: Callable[[np.ndarray], np.ndarray]  # raw model output [1,1000] -> float32 [1000]


def _float_logits(y: np.ndarray) -> np.ndarray:
    return np.asarray(y[0], dtype=np.float32)


NORMALIZED_FLOAT = IOAdapter(
    "normalized_float32",
    {"input": "float32 NCHW, ImageNet mean/std normalized", "output": "float32 logits"},
    normalize, _float_logits)

UNIT_FLOAT = IOAdapter(
    "unit_float32",
    {"input": "float32 NCHW RGB in [0, 1] (model normalizes internally)",
     "output": "float32 logits"},
    lambda x: x, _float_logits)


def _qparams(model: onnx.ModelProto, node: onnx.NodeProto) -> tuple[float, int]:
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    scale = float(numpy_helper.to_array(inits[node.input[1]]).reshape(()))
    zp = int(numpy_helper.to_array(inits[node.input[2]]).reshape(())) if len(node.input) > 2 else 0
    return scale, zp


def uint8_io_adapter(model_path: Path) -> IOAdapter:
    """Adapter for a model with a quantized uint8 image input and uint8 logits.

    The input scale/zero point are read from the DequantizeLinear that consumes
    the graph input, the output ones from the QuantizeLinear that produces the
    graph output. The [0, 1] image is quantized exactly as ONNX QuantizeLinear
    does (round half to even, saturate), and the output is dequantized.
    """
    model = onnx.load(str(model_path), load_external_data=False)
    g = model.graph
    in_name, out_name = g.input[0].name, g.output[0].name
    dq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == in_name)
    q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.output[0] == out_name)
    in_scale, in_zp = _qparams(model, dq)
    out_scale, out_zp = _qparams(model, q)
    in_scale32 = np.float32(in_scale)

    def make_input(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(x / in_scale32) + in_zp, 0, 255).astype(np.uint8)

    def to_logits(y: np.ndarray) -> np.ndarray:
        return (y[0].astype(np.float32) - np.float32(out_zp)) * np.float32(out_scale)

    return IOAdapter(
        "uint8_image_in_uint8_logits_out",
        {"input": f"uint8 NCHW RGB, q = round([0,1] image / {in_scale}) + {in_zp} "
                  "(model normalizes internally)",
         "output": f"uint8 logits, dequantized as (q - {out_zp}) * {out_scale}",
         "input_scale": in_scale, "input_zero_point": in_zp,
         "output_scale": out_scale, "output_zero_point": out_zp},
        make_input, to_logits)


# --------------------------------------------------------------------------- configs + stats


@dataclass
class RunConfig:
    key: str
    ep: str  # "cpu" or "npu"
    model: Path
    adapter: IOAdapter = NORMALIZED_FLOAT
    qnn_options: dict = field(default_factory=dict)  # added to the base QNN provider options
    cpu_reference: str | None = None  # key of the same model on the CPU EP

    def __post_init__(self) -> None:
        if self.ep not in ("cpu", "npu"):
            raise ValueError(f"ep must be 'cpu' or 'npu', got {self.ep!r}")


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile, as in docs/results-schema.md."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def latency_stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    mean = sum(s) / len(s)
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / (len(s) - 1)) if len(s) > 1 else 0.0
    return {"count": len(s), "mean_ms": mean, "p50_ms": percentile(s, 50),
            "p95_ms": percentile(s, 95), "min_ms": s[0], "max_ms": s[-1], "stddev_ms": std}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def summarize_timing(rounds: list[dict]) -> dict:
    """Medians over all rounds (init: over every session creation)."""
    ok = [r for r in rounds if "steady_state" in r]
    summary = {"rounds": len(rounds), "failed_rounds": len(rounds) - len(ok)}
    if not ok:
        return summary
    cold = [x for r in ok for x in r["cold_init_ms"]]
    cached = [x for r in ok for x in r.get("cached_init_ms", [])]
    summary["cold_init_ms_median"] = _median(cold)
    summary["cached_init_ms_median"] = _median(cached)
    for stat in ("mean_ms", "p50_ms", "p95_ms", "min_ms"):
        per_round = [r["steady_state"][stat] for r in ok]
        summary[f"steady_{stat}_median"] = _median(per_round)
        summary[f"steady_{stat}_per_round"] = per_round
    return summary


# --------------------------------------------------------------------------- harness


class NpuHarness:
    def __init__(self, log_dir: Path, cache_dir: Path, *, warmup: int = 20, runs: int = 200,
                 init_reps: int = 3, cooldown: float = 5.0, cpu_threads: int = 4,
                 split: str = "eval") -> None:
        self.log_dir, self.cache_dir = log_dir, cache_dir
        self.warmup, self.runs, self.init_reps = warmup, runs, init_reps
        self.cooldown, self.cpu_threads, self.split = cooldown, cpu_threads, split
        log_dir.mkdir(parents=True, exist_ok=False)  # one directory per run, never reused
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._inputs: dict[str, list[np.ndarray]] = {}
        self.timing_order: list[str] = []

    # ---------------------------------------------------------------- setup

    def setup(self) -> None:
        """Record power state, register the QNN EP and select the NPU device."""
        self.power_before = ort_qnn.power_state()
        print(f"power: {self.power_before}")
        if self.power_before["power_source"] != "ac":
            print("WARNING: not plugged in; timing numbers will not be comparable")
        self.plugin_path = ort_qnn.register_qnn()
        self.devices = ort_qnn.list_devices()
        print("\nEP devices:")
        for d in self.devices:
            info = ort_qnn.describe_device(d)
            print(f"  {info['ep_name']:<22} ep_vendor={info['ep_vendor']:<10} "
                  f"type={info['device_type']:<4} vendor={info['device_vendor']!r} "
                  f"vendor_id=0x{info['device_vendor_id']:x} device_id=0x{info['device_id']:x} "
                  f"metadata={info['device_metadata']}")
        self.npu = ort_qnn.select_qnn_npu(self.devices)  # raises NoNpuDeviceError
        print(f"selected NPU device: {ort_qnn.describe_device(self.npu)['device_metadata']}")
        self.base_qnn_options = {"backend_path": onnxruntime_qnn.get_qnn_htp_path(),
                                 "htp_performance_mode": HTP_PERFORMANCE_MODE}

        self.images = read_split(self.split)
        self.labels = acc.labels_for(self.images)
        print(f"\npreprocessing {len(self.images)} {self.split} images")
        self._unit = [load_image(IMAGENETTE_ROOT / p, normalized=False) for p in self.images]

    def qnn_options(self, cfg: RunConfig) -> dict:
        return {**self.base_qnn_options, **cfg.qnn_options}

    def inputs(self, adapter: IOAdapter) -> list[np.ndarray]:
        if adapter.name not in self._inputs:
            self._inputs[adapter.name] = [adapter.make_input(x) for x in self._unit]
        return self._inputs[adapter.name]

    # ---------------------------------------------------------------- sessions

    def run_all(self, session, cfg: RunConfig) -> np.ndarray:
        name = session.get_inputs()[0].name
        return np.stack([cfg.adapter.to_logits(session.run(None, {name: x})[0])
                         for x in self.inputs(cfg.adapter)])

    def cpu_session(self, model: Path, **options) -> ort.InferenceSession:
        so = ort.SessionOptions()
        for k, v in options.items():
            setattr(so, k, v)
        return ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])

    def npu_session(self, cfg: RunConfig, *, model: Path | None = None, config: dict | None = None,
                    log_stem: str) -> tuple:
        """Quiet strict NPU session: (session or None, error or None, log path).

        Only session creation is captured to the log; at error severity nothing is
        logged per Run().
        """
        so = ort_qnn.qnn_npu_session_options(self.npu, self.qnn_options(cfg), strict=True,
                                             config=config)
        log = self.log_dir / f"{log_stem}.log"
        with ort_qnn.capture_native_stderr(log):
            try:
                return ort.InferenceSession(str(model or cfg.model), sess_options=so), None, log
            except Exception as exc:  # noqa: BLE001 - recorded and reported
                return None, f"{type(exc).__name__}: {exc}", log

    @staticmethod
    def _require(result: tuple):
        session, error, log = result
        if session is None:
            raise RuntimeError(f"{error} (log: {log.name})")
        return session

    def profile_providers(self, session, feed_value: np.ndarray) -> dict:
        """Run one inference with profiling on and count executed nodes per EP."""
        session.run(None, {session.get_inputs()[0].name: feed_value})
        path = Path(session.end_profiling())
        events = json.loads(path.read_text(encoding="utf-8"))
        by_ep: dict[str, int] = {}
        nodes: dict[str, str] = {}
        for e in events:
            if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
                ep = e.get("args", {}).get("provider", "?")
                node = e["name"][: -len("_kernel_time")]
                if node not in nodes:
                    nodes[node] = ep
                    by_ep[ep] = by_ep.get(ep, 0) + 1
        return {"profile_file": path.name, "executed_nodes_by_ep": by_ep, "executed_nodes": nodes}

    def verbose_probe(self, cfg: RunConfig, stem: str, *, model: Path | None = None,
                      strict: bool = True, profile: bool = True) -> dict:
        """Create a verbose session, inspect it and release it, all while capturing its log.

        A verbose session logs on every Run(), so it is never used for accuracy or timing.
        """
        so = ort_qnn.qnn_npu_session_options(self.npu, self.qnn_options(cfg), strict=strict,
                                             log_severity=0)
        if profile:
            so.enable_profiling = True
            so.profile_file_prefix = str(self.log_dir / f"{stem}_profile")
        log = self.log_dir / f"{stem}.log"
        out: dict = {"log_file": log.name}
        with ort_qnn.capture_native_stderr(log):
            try:
                session = ort.InferenceSession(str(model or cfg.model), sess_options=so)
            except Exception as exc:  # noqa: BLE001 - recorded and reported
                session, out["error"] = None, f"{type(exc).__name__}: {exc}"
            if session is not None:
                # Snapshot while the session is alive: the HTP backend DLLs unload with it.
                out["loaded_modules"] = ort_qnn.loaded_modules(DLL_FILTER)
                if profile:
                    out["profile"] = self.profile_providers(session, self.inputs(cfg.adapter)[0])
            out["session_created"] = session is not None
            session = None
            gc.collect()
        report = ort_qnn.parse_placement_log(log.read_text(encoding="utf-8", errors="replace"))
        out.update(report.as_dict())
        out["all_nodes_on_qnn"] = bool(
            out["session_created"] and report.all_on_qnn
            and (not profile or set(out["profile"]["executed_nodes_by_ep"]) == {ort_qnn.QNN_EP}))
        return out

    def place(self, cfg: RunConfig) -> dict:
        """Strict placement report; on failure, a non-strict diagnosis of what falls back."""
        result = self.verbose_probe(cfg, f"{cfg.key}_placement")
        result["strict_session_created"] = result.pop("session_created")
        result["strict_error"] = result.pop("error", None)
        if not result["strict_session_created"]:
            result["non_strict_diagnosis"] = self.verbose_probe(
                cfg, f"{cfg.key}_placement_nonstrict", strict=False, profile=False)
        return result

    def npu_logits(self, cfg: RunConfig, stem: str) -> tuple:
        """Quiet strict session over the split: (logits or None, error or None)."""
        session, error, _ = self.npu_session(cfg, log_stem=stem)
        if session is None:
            return None, error
        out = self.run_all(session, cfg)
        session = None
        gc.collect()
        return out, None

    # ---------------------------------------------------------------- accuracy

    def evaluate(self, configs: list[RunConfig], reference: str) -> tuple[dict, dict]:
        """Placement (NPU) + accuracy for every config; agreement with ``reference``."""
        results: dict[str, dict] = {}
        logits: dict[str, np.ndarray] = {}
        for cfg in configs:
            entry = {"execution_provider": "CPUExecutionProvider" if cfg.ep == "cpu" else ort_qnn.QNN_EP,
                     "model": cfg.model.relative_to(REPO_ROOT).as_posix(),
                     "io_adapter": {"name": cfg.adapter.name, **cfg.adapter.description},
                     "cpu_reference": cfg.cpu_reference}
            results[cfg.key] = entry
            print(f"\n[{cfg.key}] {entry['model']}")
            if cfg.ep == "cpu":
                session = self.cpu_session(cfg.model)
                logits[cfg.key] = self.run_all(session, cfg)
                session = None
            else:
                entry["provider_options"] = self.qnn_options(cfg)
                entry["session_config"] = dict(ort_qnn.STRICT_CONFIG)
                entry["placement"] = p = self.place(cfg)
                self.print_placement(p)
                if not p["strict_session_created"]:
                    continue
                out, error = self.npu_logits(cfg, f"{cfg.key}_accuracy")
                if out is None:
                    entry["accuracy_error"] = error
                    print(f"  accuracy session FAILED: {error}")
                    continue
                logits[cfg.key] = out
            entry["accuracy"] = acc.accuracy(logits[cfg.key], self.labels)
            gc.collect()

        for key, entry in results.items():
            if key not in logits:
                continue
            entry["agreement_with_reference"] = (
                None if key == reference or reference not in logits
                else acc.agreement(logits[key], logits[reference], self.labels))
            ref = entry["cpu_reference"]
            if ref and ref in logits:
                entry["agreement_with_cpu_same_model"] = acc.agreement(logits[key], logits[ref],
                                                                       self.labels)
                entry["max_abs_logit_diff_vs_cpu_same_model"] = acc.max_abs_diff(logits[key],
                                                                                 logits[ref])
        self.reference = reference
        return results, logits

    @staticmethod
    def print_placement(p: dict) -> None:
        passes = p["qnn_get_capability_passes"]
        final = passes[-1] if passes else {}
        print(f"  strict session: {'OK' if p['strict_session_created'] else 'FAILED'}"
              + (f" ({p['strict_error']})" if p.get("strict_error") else ""))
        print(f"  ORT placement: {p['ort_node_placements']}; QNN validation (final pass): "
              f"{final.get('passed_nodes')} nodes passed, {final.get('failed_nodes')} failed"
              + (f"; executed: {p['profile']['executed_nodes_by_ep']}" if "profile" in p else ""))
        if p["qnn_rejections"]:
            print("  rejections:\n    " + "\n    ".join(p["qnn_rejections"][:20]))

    # ---------------------------------------------------------------- timing

    def _timed_create(self, factory) -> tuple:
        gc.collect()
        ws0 = ort_qnn.working_set_bytes()
        t0 = time.perf_counter()
        session = factory()
        t1 = time.perf_counter()
        return session, (t1 - t0) * 1000, ort_qnn.working_set_bytes() - ws0

    def _steady_state(self, session, cfg: RunConfig) -> dict:
        feed = {session.get_inputs()[0].name: self.inputs(cfg.adapter)[0]}
        for _ in range(self.warmup):
            session.run(None, feed)
        samples = []
        for _ in range(self.runs):
            t0 = time.perf_counter_ns()
            session.run(None, feed)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        return latency_stats(samples)

    def time_one(self, cfg: RunConfig, reference_logits: np.ndarray | None, tag: str) -> dict:
        """Cold init + steady state; for NPU also EP context generation + cached init."""
        timing: dict = {}
        try:
            cold_ms, cold_ws, session = [], [], None
            for rep in range(self.init_reps):
                session = None
                gc.collect()
                if cfg.ep == "cpu":
                    session, ms, ws = self._timed_create(lambda: self.cpu_session(
                        cfg.model, intra_op_num_threads=self.cpu_threads))
                else:
                    session, ms, ws = self._timed_create(lambda: self._require(self.npu_session(
                        cfg, log_stem=f"{tag}_cold_{rep}")))
                cold_ms.append(ms)
                cold_ws.append(ws)
            timing["cold_init_ms"] = cold_ms
            timing["cold_init_working_set_delta_bytes"] = cold_ws
            print(f"  cold init (ms): {[round(x, 1) for x in cold_ms]}")
            timing["steady_state"] = s = self._steady_state(session, cfg)
            print(f"  steady state: mean {s['mean_ms']:.3f} ms, p50 {s['p50_ms']:.3f}, "
                  f"p95 {s['p95_ms']:.3f}, min {s['min_ms']:.3f}")
            session = None
            if cfg.ep == "npu":
                timing.update(self._time_cached(cfg, reference_logits, tag))
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            timing["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {timing['error']}")
        gc.collect()
        return timing

    def _time_cached(self, cfg: RunConfig, reference_logits: np.ndarray | None, tag: str) -> dict:
        ctx_path = self.cache_dir / f"{cfg.key}_ctx.onnx"
        for old in self.cache_dir.glob(f"{cfg.key}_ctx*"):
            old.unlink()
        ctx_config = {"ep.context_enable": "1", "ep.context_file_path": str(ctx_path)}
        _, gen_ms, _ = self._timed_create(lambda: self._require(self.npu_session(
            cfg, config=ctx_config, log_stem=f"{tag}_ctx_generate")))
        gc.collect()
        ctx_model = onnx.load(str(ctx_path), load_external_data=False)
        ctx_files = sorted(p.name for p in self.cache_dir.glob(f"{cfg.key}_ctx*"))
        out: dict = {"ep_context": {
            "session_config": {**ort_qnn.STRICT_CONFIG, "ep.context_enable": "1",
                               "ep.context_file_path": ctx_path.relative_to(REPO_ROOT).as_posix()},
            "generation_session_ms": gen_ms,
            "files": {f: (self.cache_dir / f).stat().st_size for f in ctx_files},
            "context_model_op_types": sorted({n.op_type for n in ctx_model.graph.node}),
            "context_model_node_count": len(ctx_model.graph.node),
        }}
        cached_ms, cached_ws, cached = [], [], None
        for rep in range(self.init_reps):
            cached = None
            gc.collect()
            cached, ms, ws = self._timed_create(lambda: self._require(self.npu_session(
                cfg, model=ctx_path, log_stem=f"{tag}_cached_{rep}")))
            cached_ms.append(ms)
            cached_ws.append(ws)
        out["cached_init_ms"] = cached_ms
        out["cached_init_working_set_delta_bytes"] = cached_ws
        cached_logits = self.run_all(cached, cfg)
        cached = None
        gc.collect()
        probe = self.verbose_probe(cfg, f"{tag}_cached_placement", model=ctx_path)
        out["cached_session"] = {
            "strict_session_created": probe["session_created"],
            "error": probe.get("error"),
            "log_file": probe["log_file"],
            "ort_node_placements": probe["ort_node_placements"],
            "executed_nodes_by_ep": probe.get("profile", {}).get("executed_nodes_by_ep"),
            "all_nodes_on_qnn": probe["all_nodes_on_qnn"],
            "identical_outputs_to_uncached": (None if reference_logits is None else
                                              bool(np.array_equal(cached_logits, reference_logits))),
            "max_abs_logit_diff_vs_uncached": (None if reference_logits is None else
                                               acc.max_abs_diff(cached_logits, reference_logits)),
        }
        cs = out["cached_session"]
        print(f"  EP context generation: {gen_ms:.1f} ms; files {ctx_files}; "
              f"context model ops {out['ep_context']['context_model_op_types']}")
        print(f"  cached init (ms): {[round(x, 1) for x in cached_ms]}; strict "
              f"{'OK' if cs['all_nodes_on_qnn'] else 'NOT all on QNN'}; "
              f"placement {cs['ort_node_placements']}; identical outputs: "
              f"{cs['identical_outputs_to_uncached']}")
        return out

    def time_configs(self, configs: list[RunConfig], order: list[str], results: dict,
                     logits: dict, rounds: int = 1) -> None:
        """Time configs in ``order``, rotated by one position per round, cooldown between each.

        With as many rounds as configs, every config is timed once in every position.
        """
        by_key = {c.key: c for c in configs}
        order = [k for k in order if k in logits]
        self.timing_order, self.timing_rounds = order, rounds
        print(f"\ntiming: {rounds} round(s), {self.init_reps} session creations per init "
              f"measurement, {self.warmup} warmup + {self.runs} timed runs, "
              f"{self.cooldown}s cooldown, order {order} (rotated each round)")
        for key in order:
            results[key]["timing_rounds"] = []
        first = True
        for r in range(rounds):
            shift = r % len(order) if order else 0
            for key in order[shift:] + order[:shift]:
                if not first:
                    time.sleep(self.cooldown)
                first = False
                print(f"\n[{key}] timing, round {r + 1}")
                t = self.time_one(by_key[key], logits.get(key), f"{key}_r{r + 1}")
                t["round"] = r + 1
                results[key]["timing_rounds"].append(t)
        for key in order:
            results[key]["timing_summary"] = summarize_timing(results[key]["timing_rounds"])

    # ---------------------------------------------------------------- QNN op profile

    def qnn_op_profile(self, cfg: RunConfig, runs: int = 20, top: int = 10) -> dict:
        """Per-op HTP cycles from the QNN EP's detailed profiler (``profiling_level=detailed``).

        Detailed profiling slows execution down, so use the numbers for where time goes
        (shares between ops, accelerator time), not for latency.
        """
        csv_path = self.log_dir / f"{cfg.key}_qnn_profile.csv"
        csv_path.unlink(missing_ok=True)
        prof_cfg = RunConfig(cfg.key, "npu", cfg.model, cfg.adapter, {
            **cfg.qnn_options, "profiling_level": "detailed",
            "profiling_file_path": str(csv_path)})
        session = self._require(self.npu_session(prof_cfg, log_stem=f"{cfg.key}_qnn_profile"))
        feed = {session.get_inputs()[0].name: self.inputs(cfg.adapter)[0]}
        for _ in range(runs):
            session.run(None, feed)
        session = None
        gc.collect()

        prof = parse_qnn_profile_csv(csv_path.read_text(encoding="utf-8", errors="replace"))
        med = prof["op_cycles_median"]
        total = sum(med.values())

        return {
            "runs": runs,
            "profile_file": csv_path.name,
            "accelerator_execute_us_median": prof["accelerator_execute_us"],
            "execute_us_medians": {k: v for k, v in prof.items()
                                   if k.endswith("_us")},
            "htp_op_count": len(med),
            "htp_op_cycles_total": total,
            "top_ops": [{"op": k.split(":OpId")[0], "cycles": v,
                         "share": v / total if total else None}
                        for k, v in sorted(med.items(), key=lambda kv: -kv[1])[:top]],
        }

    # ---------------------------------------------------------------- output

    def environment(self, results: dict) -> dict:
        dlls = sorted({m for r in results.values()
                       for m in r.get("placement", {}).get("loaded_modules", [])}, key=str.lower)
        return {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "purpose": "feasibility check; latency numbers are a sanity check, not a benchmark",
            "versions": {
                "python": platform.python_version(),
                "onnxruntime": ort.__version__,
                "onnxruntime_qnn": next((d.ep_metadata.get("version") for d in self.devices
                                         if d.ep_name == ort_qnn.QNN_EP), None),
                "numpy": np.__version__,
                "onnx": onnx.__version__,
            },
            "machine": {"processor": platform.processor(), "platform": platform.platform()},
            "power_before": self.power_before,
            "power_after": ort_qnn.power_state(),
            "dll_paths": {
                "qnn_plugin_registered": self.plugin_path,
                "qnn_htp_backend": self.base_qnn_options["backend_path"],
                "onnxruntime_core": [m for m in dlls
                                     if re.search(r"onnxruntime(\.dll|_pybind11_state)", m, re.I)],
                "loaded_modules_with_npu_session": dlls,
            },
            "ep_devices": [ort_qnn.describe_device(d) for d in self.devices],
            "selected_npu_device": ort_qnn.describe_device(self.npu),
            "qnn_provider_options_base": self.base_qnn_options,
            "strict_session_config": ort_qnn.STRICT_CONFIG,
            "dataset": {"split": self.split, "count": len(self.images),
                        "images_from_imagenet_val": self.labels.from_imagenet_val,
                        "note": acc.DATASET_NOTE},
            "metrics": acc.METRIC_DEFINITIONS,
            "accuracy_reference": getattr(self, "reference", None),
            "timing_method": {
                "init": "wall time of InferenceSession construction, repeated init_reps times",
                "steady_state": "single-image Run() on a fixed preprocessed eval image, "
                                "time.perf_counter_ns around each call",
                "warmup_runs": self.warmup, "timed_runs": self.runs, "init_reps": self.init_reps,
                "cooldown_s": self.cooldown, "order": self.timing_order,
                "rounds": getattr(self, "timing_rounds", 0),
                "round_order": "order rotated left by one position per round",
                "summary": "medians over rounds (init: over all session creations)",
                "percentiles": "nearest-rank",
                "cpu_timing_session_options": {"intra_op_num_threads": self.cpu_threads},
                "cpu_accuracy_session_options": "defaults",
            },
        }

    def print_tables(self, results: dict) -> None:
        print(f"\n== Accuracy ({len(self.images)} {self.split} images; reference "
              f"{getattr(self, 'reference', None)}) ==")
        print(f"{'config':<20} {'top1/1000':>9} {'top1/10':>8} {'agreeRef':>9} {'agreeCPU':>9} "
              f"{'maxdiffCPU':>10}")
        for key, r in results.items():
            if "accuracy" not in r:
                print(f"{key:<20} {'FAILED':>9}")
                continue
            a = r["accuracy"]
            f = r.get("agreement_with_reference")
            c = r.get("agreement_with_cpu_same_model")
            ref = f"{f['top1_1000']:.3f}" if f else "-"
            cpu = f"{c['top1_1000']:.3f}" if c else "-"
            diff = f"{r['max_abs_logit_diff_vs_cpu_same_model']:.4f}" if c else "-"
            print(f"{key:<20} {a['top1_1000']:>9.3f} {a['top1_imagenette10']:>8.3f} {ref:>9} "
                  f"{cpu:>9} {diff:>10}")
        print("\n== Timing (sanity check, not a benchmark; medians over rounds) ==")
        print(f"{'config':<20} {'cold init ms':>13} {'cached init ms':>15} {'mean':>8} {'p50':>8} "
              f"{'p95':>8} {'min':>8}")
        for key in self.timing_order:
            t = results[key].get("timing_summary", {})
            if "steady_p50_ms_median" not in t:
                print(f"{key:<20} FAILED")
                continue
            cached = t["cached_init_ms_median"]
            print(f"{key:<20} {t['cold_init_ms_median']:>13.1f} "
                  f"{(f'{cached:.1f}' if cached is not None else '-'):>15} "
                  f"{t['steady_mean_ms_median']:>8.3f} {t['steady_p50_ms_median']:>8.3f} "
                  f"{t['steady_p95_ms_median']:>8.3f} {t['steady_min_ms_median']:>8.3f}")
        p = self.power_before
        print(f"\npower: {p.get('power_source')}, mode {p.get('power_mode')}, "
              f"plan {p.get('power_plan')}")
