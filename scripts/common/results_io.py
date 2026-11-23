"""Per-run output files for the measurement scripts.

Every run writes a new, timestamped summary and never overwrites an earlier
one:

    <dir>/<name>_<stamp>.json     summary of this run
    <dir>/<name>_<stamp>/         this run's logs, profiles, probe output
    <dir>/<name>_latest.json      copy of the most recent summary

``<stamp>`` is the UTC time with milliseconds, ``YYYYmmddTHHMMSS.mmmZ`` (e.g.
``20261121T235210.123Z``), the same string the C++ engine uses in its result
file names (``file_stamp()`` in ``engine/include/edgebench/run_identity.hpp``).
Files written by older versions used ``YYYYmmdd-HHMMSS``; nothing parses the
stamp, so both forms can sit side by side.

``<name>_latest.json`` is the only file that is replaced; it exists so that
docs and other scripts can refer to "the most recent run" by a fixed name.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FEASIBILITY_DIR = REPO_ROOT / "benchmarks" / "results" / "local" / "feasibility"


def run_stamp(now: datetime | None = None) -> str:
    """UTC timestamp used in file names, e.g. ``20261121T013000.123Z``.

    Milliseconds are truncated, never rounded, so ``.9999`` stays in its second.
    A naive ``now`` is taken as UTC.
    """
    t = now or datetime.now(timezone.utc)
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc)
    return f"{t:%Y%m%dT%H%M%S}.{t.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class RunOutput:
    directory: Path
    name: str
    stamp: str

    @property
    def summary(self) -> Path:
        return self.directory / f"{self.name}_{self.stamp}.json"

    @property
    def run_dir(self) -> Path:
        """Directory for this run's logs and other files (created on first use)."""
        return self.directory / f"{self.name}_{self.stamp}"

    @property
    def latest(self) -> Path:
        return self.directory / f"{self.name}_latest.json"

    def make_run_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        return self.run_dir

    def write(self, data: dict) -> Path:
        """Write the summary (refusing to overwrite) and refresh ``<name>_latest.json``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2) + "\n"
        with self.summary.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
        shutil.copyfile(self.summary, self.latest)
        return self.summary


def new_run(directory: Path, name: str, now: datetime | None = None) -> RunOutput:
    """A RunOutput whose summary file and run directory do not exist yet."""
    out = RunOutput(Path(directory).resolve(), name, run_stamp(now))
    if out.summary.exists() or out.run_dir.exists():
        raise FileExistsError(f"{out.summary} already exists; refusing to overwrite")
    return out
