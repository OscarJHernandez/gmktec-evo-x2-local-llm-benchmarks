"""Portable system information collection and optional GPU collector interface."""

from __future__ import annotations

import platform
from collections.abc import Callable

import psutil

from llm_bench.models import HardwareInfo
from llm_bench.telemetry.discovery import cpu_name, discover_gpus

GpuCollector = Callable[[], tuple[str | None, int | None, str | None]]


def collect_hardware(gpu_collector: GpuCollector | None = None) -> HardwareInfo:
    try:
        memory = psutil.virtual_memory()
    except (OSError, RuntimeError, psutil.Error):
        memory = None
    try:
        cpu_count = psutil.cpu_count()
    except (OSError, RuntimeError, psutil.Error):
        cpu_count = None
    gpu_name: str | None = None
    gpu_bytes: int | None = None
    gpu_source: str | None = None
    try:
        devices = discover_gpus()
    except (OSError, RuntimeError):
        devices = []
    if devices:
        gpu_name = "; ".join(device["name"] for device in devices)
    if gpu_collector is not None:
        try:
            gpu_name, gpu_bytes, gpu_source = gpu_collector()
            if (
                (gpu_name is not None and not isinstance(gpu_name, str))
                or (
                    gpu_bytes is not None
                    and (
                        isinstance(gpu_bytes, bool)
                        or not isinstance(gpu_bytes, int)
                        or gpu_bytes < 0
                    )
                )
                or (gpu_source is not None and not isinstance(gpu_source, str))
            ):
                raise ValueError("GPU collector returned invalid values")
        except (OSError, RuntimeError, TypeError, ValueError):
            gpu_name, gpu_bytes, gpu_source = None, None, None
    return HardwareInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu=cpu_name(),
        logical_cpu_count=cpu_count,
        total_ram_bytes=memory.total if memory else None,
        available_ram_bytes=memory.available if memory else None,
        ram_percent=memory.percent if memory else None,
        gpu=gpu_name,
        gpu_memory_bytes=gpu_bytes,
        gpu_memory_source=gpu_source,
    )
