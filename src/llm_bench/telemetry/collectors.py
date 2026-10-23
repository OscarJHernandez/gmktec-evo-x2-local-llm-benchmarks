"""Capability-probed, non-blocking psutil/sysfs collectors and optional SMI fallback."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import psutil

from llm_bench.telemetry.discovery import discover_gpus, text_file
from llm_bench.telemetry.schemas import Capability

# Canonical units and numeric representation. Counter metrics remain integers.
REGISTRY: dict[str, tuple[str, Literal["int", "float", "bool"]]] = {
    "cpu_utilization": ("percent", "float"),
    "cpu_frequency": ("Hz", "float"),
    "gpu_utilization": ("percent", "float"),
    "gpu_frequency": ("Hz", "float"),
    "gpu_memory_frequency": ("Hz", "float"),
    "ram_total": ("bytes", "int"),
    "ram_available": ("bytes", "int"),
    "ram_used": ("bytes", "int"),
    "swap_used": ("bytes", "int"),
    "vram_used": ("bytes", "int"),
    "vram_capacity": ("bytes", "int"),
    "gtt_used": ("bytes", "int"),
    "gtt_capacity": ("bytes", "int"),
    "temperature": ("degC", "float"),
    "fan_speed": ("rpm", "float"),
    "power": ("W", "float"),
    "energy": ("J", "float"),
    "throttled": ("boolean", "bool"),
    "memory_bandwidth": ("bytes/s", "float"),
    "load_1m": ("processes", "float"),
    "disk_read": ("bytes", "int"),
    "disk_written": ("bytes", "int"),
    "network_received": ("bytes", "int"),
    "network_sent": ("bytes", "int"),
    "process_cpu": ("percent", "float"),
    "process_rss": ("bytes", "int"),
    "process_threads": ("threads", "int"),
}


def _attribute(reader: Callable[[], Any], name: str) -> Any:
    return getattr(reader(), name)


def capability(metric: str, source: str, **kwargs: Any) -> Capability:
    unit, dtype = REGISTRY[metric]
    return Capability(metric=metric, unit=unit, source=source, value_type=dtype, **kwargs)


def parse_smi(payload: Any, tool: str) -> dict[str, dict[str, float]]:
    """Parse explicitly recognized JSON fields; never coerce N/A to zero.

    Device mapping uses PCI BDF, not an assumed GPU enumeration order.
    """
    result: dict[str, dict[str, float]] = {}
    cards = (
        payload.get("gpu_data", []) if tool == "amd-smi" and isinstance(payload, dict) else payload
    )
    if isinstance(cards, dict):
        cards = list(cards.values())
    if not isinstance(cards, list):
        return result
    fields = {
        "GPU use (%)": ("gpu_utilization", 1.0),
        "GPU Memory Allocated (VRAM%)": ("ignored_allocation_percent", 1.0),
        "VRAM Total Used Memory (B)": ("vram_used", 1.0),
        "GTT Total Used Memory (B)": ("gtt_used", 1.0),
    }
    for card in cards:
        if not isinstance(card, dict):
            continue
        bdf = card.get("PCI Bus") or card.get("bdf")
        if not isinstance(bdf, str):
            continue
        metrics: dict[str, float] = {}
        for key, (metric, scale) in fields.items():
            raw = card.get(key)
            if not isinstance(raw, str | int | float) or isinstance(raw, bool):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if metric in REGISTRY and math.isfinite(value) and value >= 0:
                if metric == "gpu_utilization" and value > 100:
                    continue
                metrics[metric] = value * scale
        # AMD SMI structured metric JSON (recognized versions only).
        usage = card.get("usage", {})
        if isinstance(usage, dict):
            raw = usage.get("gfx_activity")
            if isinstance(raw, int | float) and not isinstance(raw, bool) and 0 <= raw <= 100:
                metrics["gpu_utilization"] = float(raw)
        result[bdf.lower()] = metrics
    return result


class Collector:
    def __init__(self, sys: Path = Path("/sys")) -> None:
        self.capabilities: list[Capability] = []
        self.readers: list[tuple[Capability, Callable[[], Any]]] = []
        self.processes: dict[tuple[int, float], Any] = {}
        self.gpus = discover_gpus(sys)
        self.smi = shutil.which("amd-smi") or shutil.which("rocm-smi")
        self.smi_disabled: str | None = None
        self._discover(sys)

    def add(self, metric: str, source: str, reader: Callable[[], Any], **kwargs: Any) -> None:
        cap = capability(metric, source, **kwargs)
        self.capabilities.append(cap)
        self.readers.append((cap, reader))

    def unavailable(self, metric: str, reason: str, **kwargs: Any) -> None:
        self.capabilities.append(
            capability(metric, "unavailable", available=False, reason=reason, **kwargs)
        )

    def _file(self, path: Path, metric: str, scale: float = 1.0, **kwargs: Any) -> None:
        def read() -> int | float:
            raw = text_file(path)
            return int(raw) if REGISTRY[metric][1] == "int" and scale == 1 else float(raw) * scale

        self.add(metric, str(path), read, **kwargs)

    def _discover(self, sys: Path) -> None:
        self.add("cpu_utilization", "psutil.cpu_percent", lambda: psutil.cpu_percent(None))
        # Prime counters on the same worker thread that will collect them.
        for index in range(psutil.cpu_count() or 0):
            self.add(
                "cpu_utilization",
                "psutil.cpu_percent(percpu)",
                lambda: 0,
                device=f"cpu:{index}",
            )
        for metric, attr in (
            ("ram_total", "total"),
            ("ram_used", "used"),
            ("ram_available", "available"),
        ):
            self.add(
                metric, "psutil.virtual_memory", partial(_attribute, psutil.virtual_memory, attr)
            )
        self.add("swap_used", "psutil.swap_memory", lambda: psutil.swap_memory().used)
        self.add("load_1m", "psutil.getloadavg", lambda: psutil.getloadavg()[0])
        for metric, attr in (("disk_read", "read_bytes"), ("disk_written", "write_bytes")):
            self.add(
                metric,
                "psutil.disk_io_counters",
                partial(_attribute, psutil.disk_io_counters, attr),
            )
        for metric, attr in (("network_received", "bytes_recv"), ("network_sent", "bytes_sent")):
            self.add(
                metric,
                "psutil.net_io_counters",
                partial(_attribute, psutil.net_io_counters, attr),
            )
        for path in sorted((sys / "devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
            self._file(path, "cpu_frequency", 1000, device=path.parent.parent.name)
        for gpu in self.gpus:
            root, device = Path(gpu["path"]), gpu["device"]
            for filename, metric in (
                ("gpu_busy_percent", "gpu_utilization"),
                ("mem_info_vram_used", "vram_used"),
                ("mem_info_vram_total", "vram_capacity"),
                ("mem_info_gtt_used", "gtt_used"),
                ("mem_info_gtt_total", "gtt_capacity"),
            ):
                path = root / filename
                if path.exists():
                    self._file(path, metric, device=device, scope="gpu")
                else:
                    self.unavailable(metric, "sysfs attribute absent", device=device, scope="gpu")
        for hwmon in sorted((sys / "class/hwmon").glob("hwmon*")):
            try:
                name = text_file(hwmon / "name")
            except OSError:
                name = "unidentified"
            physical = (hwmon / "device").resolve()
            device = next(
                (g["device"] for g in self.gpus if Path(g["path"]) == physical),
                f"{name}:{physical.name}",
            )
            for pattern, metric, scale in (
                ("temp*_input", "temperature", 0.001),
                ("fan*_input", "fan_speed", 1),
                ("freq*_input", "gpu_frequency", 1),
                ("power*_input", "power", 1e-6),
                ("power*_average", "power", 1e-6),
                ("energy*_input", "energy", 1e-6),
            ):
                for path in sorted(hwmon.glob(pattern)):
                    try:
                        label = text_file(path.with_name(path.name.rsplit("_", 1)[0] + "_label"))
                    except OSError:
                        label = path.stem
                    # amdgpu power can include CPU on APUs; conservative SoC scope.
                    scope = (
                        "package_or_soc"
                        if name == "amdgpu" and metric in {"power", "energy"}
                        else name
                    )
                    actual = (
                        "gpu_memory_frequency"
                        if name == "amdgpu" and path.name == "freq2_input"
                        else metric
                    )
                    if metric == "gpu_frequency" and name != "amdgpu":
                        continue
                    self._file(path, actual, scale, device=device, scope=scope, label=label)
        energy_paths: set[Path] = set()
        for zone in (sys / "class/powercap").glob("*"):
            # Class entries are commonly symlinks; recursive glob alone skips them.
            physical_zone = zone.resolve()
            energy_paths.update(path.resolve() for path in physical_zone.glob("**/energy_uj"))
        for root in sorted(energy_paths):
            try:
                label = text_file(root.parent / "name")
                counter_range = float(text_file(root.parent / "max_energy_range_uj")) * 1e-6
            except (OSError, ValueError):
                label, counter_range = root.parent.name, None
            if counter_range is not None and counter_range <= 0:
                counter_range = None
            self._file(
                root,
                "energy",
                1e-6,
                device=root.parent.name,
                scope="package",
                label=label,
                counter_range=counter_range,
            )
        for metric in (
            "temperature",
            "fan_speed",
            "power",
            "energy",
            "cpu_frequency",
            "gpu_utilization",
            "vram_used",
            "gtt_used",
            "memory_bandwidth",
            "throttled",
        ):
            if not any(c.metric == metric for c in self.capabilities):
                self.unavailable(metric, "no reliable readable interface discovered")

    def prime(self) -> None:
        psutil.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)

    def read(self) -> list[tuple[Capability, Any, str | None]]:
        rows: list[tuple[Capability, Any, str | None]] = []
        # One per-core call per tick: repeated calls would reset its baseline.
        try:
            per_core = psutil.cpu_percent(None, percpu=True)
        except (OSError, RuntimeError):
            per_core = []
        for cap, read in self.readers:
            try:
                value = (
                    per_core[int(cap.device.split(":")[1])]
                    if cap.source.endswith("(percpu)")
                    else read()
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or (value < 0 and cap.metric != "temperature")
                ):
                    raise ValueError("invalid numeric sensor value")
                if cap.unit == "percent" and value > 100:
                    raise ValueError("percentage outside 0..100")
                rows.append((cap, value, None))
            except (
                OSError,
                ValueError,
                IndexError,
                AttributeError,
                RuntimeError,
                psutil.Error,
            ) as exc:
                rows.append((cap, None, type(exc).__name__))
        rows.extend(self._read_processes())
        rows.extend(self._read_smi())
        return rows

    def _read_processes(self) -> list[tuple[Capability, Any, str | None]]:
        rows: list[tuple[Capability, Any, str | None]] = []
        alive: dict[tuple[int, float], Any] = {}
        try:
            candidates = list(psutil.process_iter(["pid", "name", "create_time", "ppid"]))
        except psutil.Error:
            return [
                (
                    capability("process_rss", "psutil", device="ollama"),
                    None,
                    "process discovery unavailable",
                )
            ]
        # Resolve descendants from one snapshot rather than rescanning /proc for
        # every Ollama parent's children on every tick.
        by_pid = {process.pid: process for process in candidates}
        children: dict[int, list[int]] = {}
        pending = []
        for process in candidates:
            parent_pid = process.info.get("ppid")
            if isinstance(parent_pid, int):
                children.setdefault(parent_pid, []).append(process.pid)
            if "ollama" in (process.info.get("name") or "").lower():
                pending.append(process.pid)
        selected: dict[int, Any] = {}
        while pending:
            pid = pending.pop()
            if pid in selected:
                continue
            selected[pid] = by_pid[pid]
            pending.extend(children.get(pid, []))
        for process in selected.values():
            try:
                name = (process.info.get("name") or "unknown").lower()
                key = (process.pid, process.create_time())
                cached = self.processes.get(key, process)
                alive[key] = cached
                device = f"pid:{key[0]}:{key[1]}"
                for metric, value in (
                    ("process_cpu", cached.cpu_percent(None) if key in self.processes else None),
                    ("process_rss", cached.memory_info().rss),
                    ("process_threads", cached.num_threads()),
                ):
                    cap = capability(metric, "psutil", device=device, scope="process", label=name)
                    rows.append(
                        (cap, value, "baseline not yet available" if value is None else None)
                    )
                if key not in self.processes:
                    cached.cpu_percent(None)
            except (psutil.Error, OSError):
                rows.append(
                    (
                        capability("process_rss", "psutil", device=f"pid:{process.pid}"),
                        None,
                        "process exited or permission denied",
                    )
                )
        self.processes = alive
        if not rows:
            rows.extend(
                (
                    capability(metric, "psutil", device="ollama", scope="process"),
                    None,
                    "no accessible Ollama process",
                )
                for metric in ("process_cpu", "process_rss", "process_threads")
            )
        return rows

    def _read_smi(self) -> list[tuple[Capability, Any, str | None]]:
        missing = [c for c in self.capabilities if not c.available and c.scope == "gpu"]
        if not self.smi or not missing or self.smi_disabled:
            return []
        tool = Path(self.smi).name
        args = (
            [self.smi, "metric", "--json"]
            if tool == "amd-smi"
            else [self.smi, "--showbus", "--showuse", "--showmeminfo", "vram", "gtt", "--json"]
        )
        try:
            output = subprocess.run(args, capture_output=True, text=True, timeout=0.5, check=True)
            parsed = parse_smi(json.loads(output.stdout), tool)
            if not parsed:
                self.smi_disabled = "unrecognized JSON layout or missing PCI BDF"
                return []
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.smi_disabled = type(exc).__name__
            return []
        rows: list[tuple[Capability, Any, str | None]] = []
        for cap in missing:
            value = parsed.get(cap.device.lower(), {}).get(cap.metric)
            if value is not None:
                fallback = cap.model_copy(
                    update={"source": tool, "available": True, "reason": None}
                )
                rows.append((fallback, value, None))
                if not any(
                    c.metric == fallback.metric
                    and c.device == fallback.device
                    and c.source == fallback.source
                    for c in self.capabilities
                ):
                    self.capabilities.append(fallback)
        if not rows:
            self.smi_disabled = "no compatible missing capabilities returned"
        return rows
