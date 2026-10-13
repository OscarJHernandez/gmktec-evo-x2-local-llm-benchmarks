"""Portable system information collection and optional GPU collector interface."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable

import psutil

from llm_bench.models import HardwareInfo

GpuCollector = Callable[[], tuple[str | None, int | None, str | None]]


def collect_hardware(gpu_collector: GpuCollector | None = None) -> HardwareInfo:
    memory = psutil.virtual_memory()
    gpu_name: str | None = None
    gpu_bytes: int | None = None
    gpu_source: str | None = None
    if gpu_collector is not None:
        try:
            gpu_name, gpu_bytes, gpu_source = gpu_collector()
        except (OSError, RuntimeError, ValueError):
            pass
    return HardwareInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu=platform.processor() or "unknown",
        logical_cpu_count=psutil.cpu_count(),
        total_ram_bytes=memory.total,
        available_ram_bytes=memory.available,
        ram_percent=memory.percent,
        gpu=gpu_name,
        gpu_memory_bytes=gpu_bytes,
        gpu_memory_source=gpu_source,
    )
