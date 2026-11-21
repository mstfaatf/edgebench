"""ONNX Runtime + QNN execution provider helpers for scripts that run on the NPU.

- registering the ``onnxruntime-qnn`` plugin EP and enumerating EP devices,
- selecting the QNN device whose hardware type is NPU (never "the first QNN
  device": the plugin also exposes the Adreno GPU),
- strict session options (``session.disable_cpu_ep_fallback=1``),
- capturing ONNX Runtime's native log output (written to the process's stderr
  file descriptor) and parsing node placement from it,
- Windows facts that affect measurements: loaded DLL paths, AC/battery state,
  power mode, and process working set.

Only importable on Windows with ``onnxruntime`` and ``onnxruntime-qnn``
installed; callers are expected to have passed the platform guard first.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import re
import sys
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import onnxruntime as ort
import onnxruntime_qnn

QNN_EP = "QNNExecutionProvider"
STRICT_CONFIG = {"session.disable_cpu_ep_fallback": "1"}


# --------------------------------------------------------------------------- devices


def register_qnn() -> str:
    """Register the QNN plugin EP library (once per process); return its path."""
    path = onnxruntime_qnn.get_library_path()
    try:
        ort.register_execution_provider_library(QNN_EP, path)
    except Exception as exc:  # noqa: BLE001 - re-registration raises; anything else is fatal
        if "already registered" not in str(exc).lower():
            raise
    return path


def describe_device(d) -> dict:
    return {
        "ep_name": d.ep_name,
        "ep_vendor": d.ep_vendor,
        "ep_metadata": dict(d.ep_metadata),
        "ep_options": dict(d.ep_options),
        "device_type": d.device.type.name,
        "device_vendor": d.device.vendor,
        "device_vendor_id": d.device.vendor_id,
        "device_id": d.device.device_id,
        "device_metadata": dict(d.device.metadata),
    }


def list_devices() -> list:
    return list(ort.get_ep_devices())


class NoNpuDeviceError(RuntimeError):
    pass


def select_qnn_npu(devices: list):
    """Return the single QNN EP device whose hardware type is NPU."""
    npus = [d for d in devices
            if d.ep_name == QNN_EP and d.device.type == ort.OrtHardwareDeviceType.NPU]
    if not npus:
        found = ", ".join(f"{d.ep_name}/{d.device.type.name}" for d in devices) or "none"
        raise NoNpuDeviceError(f"no {QNN_EP} device of type NPU found (devices: {found})")
    if len(npus) > 1:
        raise NoNpuDeviceError(f"expected one QNN NPU device, found {len(npus)}")
    return npus[0]


def qnn_npu_session_options(npu_device, provider_options: dict, *, strict: bool = True,
                            config: dict | None = None, log_severity: int = 3) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.log_severity_level = log_severity
    so.log_verbosity_level = 0
    for k, v in {**(STRICT_CONFIG if strict else {}), **(config or {})}.items():
        so.add_session_config_entry(k, v)
    so.add_provider_for_devices([npu_device], provider_options)
    return so


# --------------------------------------------------------------------------- native logs


@contextmanager
def capture_native_stderr(path: Path):
    """Redirect the process-level stderr file descriptor (ORT's log sink) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.flush()
    saved = os.dup(2)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(fd, 2)
        yield path
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(fd)
        os.close(saved)


_LOG_LINE = re.compile(r"\[(?P<sev>[VIWEF]):onnxruntime:[^,]*, (?P<src>[\w.]+):\d+ [^\]]*\] (?P<msg>.*)")
_PASS = re.compile(r"GetCapability pass (?P<n>\d+)")
_VALID = re.compile(r"Validation (?P<res>PASSED|FAILED) for (?P<n>\d+) nodes in OrtNodeUnit \((?P<op>\w+)\)")
_PLACED = re.compile(r"(?:All nodes|Node\(s\)) placed on \[(?P<ep>\w+)\]\. Number of nodes: (?P<n>\d+)")
_REJECT_HINT = re.compile(r"not supported|unsupported|FAILED|cannot|failed", re.IGNORECASE)


@dataclass
class PlacementReport:
    ort_placements: dict = field(default_factory=dict)  # EP -> nodes (after EP fusion)
    qnn_passes: list = field(default_factory=list)  # per GetCapability pass
    rejections: list = field(default_factory=list)  # EP log lines explaining rejected nodes
    warnings: list = field(default_factory=list)  # distinct W/E lines

    def as_dict(self) -> dict:
        return {
            "ort_node_placements": self.ort_placements,
            "qnn_get_capability_passes": self.qnn_passes,
            "qnn_rejections": self.rejections,
            "log_warnings": self.warnings,
        }

    @property
    def all_on_qnn(self) -> bool:
        final = self.qnn_passes[-1] if self.qnn_passes else None
        return (set(self.ort_placements) == {QNN_EP}
                and final is not None and final["failed_nodes"] == 0)


def parse_placement_log(text: str) -> PlacementReport:
    """Extract node placement facts from a verbose (severity 0) ORT session log."""
    report = PlacementReport()
    current = None
    warnings: list[str] = []
    for line in text.splitlines():
        m = _LOG_LINE.search(line)
        if not m:
            continue
        msg, src, sev = m["msg"].strip(), m["src"], m["sev"]
        if p := _PASS.search(msg):
            current = {"pass": int(p["n"]), "passed_units": Counter(), "passed_nodes": 0,
                       "failed_units": Counter(), "failed_nodes": 0}
            report.qnn_passes.append(current)
        elif (v := _VALID.search(msg)) and current is not None:
            key = "passed" if v["res"] == "PASSED" else "failed"
            current[f"{key}_units"][v["op"]] += 1
            current[f"{key}_nodes"] += int(v["n"])
            if key == "failed":
                report.rejections.append(msg)
        elif pl := _PLACED.search(msg):
            report.ort_placements[pl["ep"]] = report.ort_placements.get(pl["ep"], 0) + int(pl["n"])
        elif src.startswith(("qnn_", "base_op_builder", "conv_op_builder")) and sev in "WE" \
                and _REJECT_HINT.search(msg):
            report.rejections.append(msg)
        if sev in "WEF" and msg not in warnings:
            warnings.append(msg)
    for p in report.qnn_passes:
        p["passed_units"] = dict(sorted(p["passed_units"].items()))
        p["failed_units"] = dict(sorted(p["failed_units"].items()))
    report.warnings = warnings
    return report


# --------------------------------------------------------------------------- Windows facts


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def working_set_bytes() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


def loaded_modules(name_filter: re.Pattern | None = None) -> list[str]:
    """Full paths of the DLLs loaded into this process (optionally filtered by file name)."""
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD,
                                         ctypes.POINTER(wt.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
    proc = kernel32.GetCurrentProcess()
    needed = wt.DWORD()
    modules = (wt.HMODULE * 2048)()
    if not psapi.EnumProcessModules(proc, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError()
    paths = []
    buf = ctypes.create_unicode_buffer(32768)
    for h in modules[: needed.value // ctypes.sizeof(wt.HMODULE)]:
        if psapi.GetModuleFileNameExW(proc, h, buf, len(buf)):
            paths.append(buf.value)
    if name_filter is not None:
        paths = [p for p in paths if name_filter.search(Path(p).name)]
    return sorted(paths, key=str.lower)


class _Guid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    def as_uuid(self) -> uuid.UUID:
        return uuid.UUID(bytes_le=bytes(self.bytes))


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wt.DWORD), ("BatteryFullLifeTime", wt.DWORD)]


# Power mode slider ("overlay") GUIDs on Windows 11; the all-zero GUID means Balanced.
POWER_MODE_OVERLAYS = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}
POWER_PLANS = {
    "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
    "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
}


def power_state() -> dict:
    """AC/battery status, the Settings power mode, and the active power plan."""
    status = _SystemPowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    ac = {0: "battery", 1: "ac", 255: "unknown"}.get(status.ACLineStatus, "unknown")
    state = {
        "power_source": ac,
        "battery_percent": None if status.BatteryLifePercent == 255 else status.BatteryLifePercent,
        "battery_saver_on": bool(status.SystemStatusFlag),
    }
    powrprof = ctypes.WinDLL("powrprof")
    overlay = _Guid()
    if powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(overlay)) == 0:
        g = str(overlay.as_uuid())
        state["power_mode"] = POWER_MODE_OVERLAYS.get(g, "unknown")
        state["power_mode_overlay_guid"] = g
    plan = ctypes.POINTER(_Guid)()
    if powrprof.PowerGetActiveScheme(None, ctypes.byref(plan)) == 0:
        g = str(plan.contents.as_uuid())
        state["power_plan"] = POWER_PLANS.get(g, "custom")
        state["power_plan_guid"] = g
        ctypes.windll.kernel32.LocalFree(plan)
    return state


def new_log_path(directory: Path | None, stem: str) -> Path:
    if directory is None:
        handle, name = tempfile.mkstemp(suffix=".log", prefix=f"{stem}_")
        os.close(handle)
        return Path(name)
    return directory / f"{stem}.log"
