"""Command-line options shared by the NPU check scripts (no heavy imports)."""

from __future__ import annotations

import argparse


def add_timing_args(parser: argparse.ArgumentParser, *, rounds: int = 1) -> None:
    parser.add_argument("--warmup", type=int, default=20, help="untimed runs before timing")
    parser.add_argument("--runs", type=int, default=200, help="timed single-image runs")
    parser.add_argument("--init-reps", type=int, default=3,
                        help="session creations per init measurement")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="seconds between timed configurations")
    parser.add_argument("--rounds", type=int, default=rounds,
                        help="timing rounds; the order is rotated by one position per round")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="intra-op threads for CPU timing sessions (0 = ORT default, one "
                             "per core); the default thread pool is very sensitive to "
                             "background load")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="directory for the timestamped summary and run files "
                             "(default: benchmarks/results/local/feasibility)")


def harness_kwargs(args: argparse.Namespace) -> dict:
    return {"warmup": args.warmup, "runs": args.runs, "init_reps": args.init_reps,
            "cooldown": args.cooldown, "cpu_threads": args.cpu_threads}
