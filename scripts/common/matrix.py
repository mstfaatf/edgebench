"""The CPU/NPU comparison matrix: which `edgebench run` configurations it holds.

Shared by ``scripts/run_matrix.py`` (runs them) and ``scripts/check_matrix.py``
(checks the results). Standard library only.

Every model of the pipeline (the three image classifiers with external
normalization, pruned INT8; the autoencoder's cleaned graph) at batch 1, 4 and
8, each in four configurations:

- ``cpu_fp32``: the FP32 file on the CPU EP;
- ``cpu_int8``: the INT8 QDQ file on the CPU EP;
- ``npu_int8``: the INT8 QDQ file on the NPU (strict, as stored);
- ``npu_fp16``: the FP32 file on the NPU with FP16 compute (``--npu-fp16``).

Plus, for the image models: NPU INT8 at batch 8 with
``--htp-graph-finalization-mode 1`` (``npu_int8_finalization1``), and the
normalization-folded INT8 file at batch 1 on the CPU and the NPU
(``cpu_int8_folded``, ``npu_int8_folded``).

NPU runs use ``--ep-cache use`` (after a ``--ep-cache build`` run of the same
key), ``--init-breakdown`` and ``--profile-accelerator``. ``--init-breakdown``
is an NPU option (the CPU provider has no EP context), so a CPU run's timed
session creations are its cold initializations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.model_registry import BATCH_SIZES, relative_path

MATRIX_VERSION = "1.0.0"
IMAGE_MODELS = ("mobilenetv2", "resnet18", "efficientnet_lite0")
TIME_SERIES_MODELS = ("conv1d_autoencoder",)
FINALIZATION_OPTION = "htp_graph_finalization_optimization_mode"


@dataclass(frozen=True)
class MatrixRun:
    run_id: str
    model: str
    config: str               # cpu_fp32 | cpu_int8 | npu_int8 | npu_fp16 | ...
    variant: str              # model variant of the file
    batch: int
    provider: str             # cpu | npu
    stored_precision: str     # fp32 | int8
    compute_precision: str    # fp32 | int8 | fp16
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    ep_compile_options: tuple[tuple[str, str], ...] = ()

    @property
    def file(self) -> str:
        return relative_path(self.model, self.variant, self.batch)

    @property
    def npu(self) -> bool:
        return self.provider == "npu"


def _run(model: str, config: str, variant: str, batch: int, extra=(), options=()) -> MatrixRun:
    provider = config.split("_")[0]
    stored = "int8" if variant.startswith("int8") else "fp32"
    compute = "fp16" if config.startswith("npu_fp16") else stored
    args = ("--npu-fp16",) if compute == "fp16" else ()
    return MatrixRun(f"{model}_b{batch}_{config}", model, config, variant, batch, provider,
                     stored, compute, args + tuple(extra), tuple(options))


def planned_runs() -> list[MatrixRun]:
    runs = []
    for model in IMAGE_MODELS + TIME_SERIES_MODELS:
        for batch in BATCH_SIZES:
            runs += [_run(model, "cpu_fp32", "fp32", batch), _run(model, "cpu_int8", "int8", batch),
                     _run(model, "npu_int8", "int8", batch), _run(model, "npu_fp16", "fp32", batch)]
    for model in IMAGE_MODELS:
        runs.append(_run(model, "npu_int8_finalization1", "int8", 8,
                         ("--htp-graph-finalization-mode", "1"), ((FINALIZATION_OPTION, "1"),)))
        runs.append(_run(model, "cpu_int8_folded", "int8_folded", 1))
        runs.append(_run(model, "npu_int8_folded", "int8_folded", 1))
    return runs


def run_arguments(run: MatrixRun, *, trials: int, iterations: int, warmup: int,
                  cooldown_ms: int, cache_dir: str) -> list[str]:
    """`edgebench run` arguments of a timed matrix run (without --model/--output-dir/--name)."""
    args = ["--provider", run.provider, "--trials", str(trials), "--iterations", str(iterations),
            "--warmup", str(warmup), "--cooldown-ms", str(cooldown_ms), *run.extra_args]
    if run.npu:
        args += ["--ep-cache", "use", "--ep-cache-dir", cache_dir, "--init-breakdown",
                 "--profile-accelerator"]
    return args


def cache_build_arguments(run: MatrixRun, cache_dir: str) -> list[str]:
    """A short `--ep-cache build` run that builds the cache entry of `run`'s key."""
    return ["--provider", "npu", "--trials", "1", "--iterations", "1", "--warmup", "0",
            "--cooldown-ms", "0", "--load-sample-ms", "0", *run.extra_args,
            "--ep-cache", "build", "--ep-cache-dir", cache_dir]
