"""Guard against running EdgeBench scripts on a non-native (emulated) interpreter.

On Windows on Arm, an x64 Python runs transparently under emulation. Every
library then loads its x64 build, and any timing or accuracy result is for the
emulated stack, not the native one. ``platform.machine()`` cannot detect this:
it reports the host OS architecture (``ARM64``) even inside an emulated x64
interpreter. ``sysconfig.get_platform()`` reports what the interpreter itself
was built for, and is what this module checks.

When to use it:

- **Guard** (call ``require_native_arm64()`` and exit on failure): scripts that
  run models, load ONNX Runtime, or measure anything about the machine, e.g.
  ``tools/ort_probe/reference.py``. Their output is only meaningful natively.
- **Report only**: ``scripts/check_env.py`` is a diagnostic; it prints the
  platform status via ``platform_status()`` / ``explain()`` instead of exiting.
- **No guard**: platform-independent tooling such as
  ``scripts/validate_results.py``, which must run anywhere (Linux/x64 CI,
  contributors on other machines). Such scripts must not import this module.

Typical use in a guarded script::

    from common.platform_guard import NonNativeInterpreterError, require_native_arm64

    try:
        require_native_arm64()
    except NonNativeInterpreterError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
"""

from __future__ import annotations

import platform
import sys
import sysconfig
from dataclasses import dataclass

EXPECTED_PLATFORM = "win-arm64"


class NonNativeInterpreterError(RuntimeError):
    """Raised when the interpreter is not a native win-arm64 build."""


@dataclass(frozen=True)
class PlatformStatus:
    interpreter_platform: str  # sysconfig.get_platform(), e.g. "win-arm64" or "win-amd64"
    host_machine: str  # platform.machine(); host OS architecture, not the interpreter's
    executable: str

    @property
    def is_native_arm64(self) -> bool:
        return self.interpreter_platform == EXPECTED_PLATFORM


def platform_status() -> PlatformStatus:
    """Return the interpreter's build platform without raising."""
    return PlatformStatus(
        interpreter_platform=sysconfig.get_platform(),
        host_machine=platform.machine(),
        executable=sys.executable,
    )


def explain(status: PlatformStatus) -> str:
    """Human-readable explanation of why a non-native interpreter is a problem."""
    emulated = status.interpreter_platform == "win-amd64" and status.host_machine.upper() == "ARM64"
    lines = [
        f"This Python interpreter is built for '{status.interpreter_platform}', "
        f"but EdgeBench requires a native '{EXPECTED_PLATFORM}' interpreter.",
        f"  interpreter: {status.executable}",
    ]
    if emulated:
        lines.append(
            "  This is an x64 Python running under emulation on an ARM64 machine. "
            "(platform.machine() still reports ARM64, so it cannot be used to detect this.)"
        )
    lines += [
        "",
        "To fix it, find the native ARM64 interpreter and create the venv with its full path:",
        "  py -0p                                  # lists installed interpreters; look for '-arm64'",
        r"  & 'C:\path\to\Python312-arm64\python.exe' -m venv .venv",
        r"  .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
        "",
        "See docs/setup.md for details.",
    ]
    return "\n".join(lines)


def require_native_arm64() -> PlatformStatus:
    """Raise NonNativeInterpreterError unless running on a native win-arm64 interpreter."""
    status = platform_status()
    if not status.is_native_arm64:
        raise NonNativeInterpreterError(explain(status))
    return status
