from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from test_core import FakeBackend

from llm_bench.models import BenchmarkConfig
from llm_bench.runner import run_benchmark
from llm_bench.telemetry.aggregation import summarize_telemetry
from llm_bench.telemetry.collectors import Collector, capability, parse_smi
from llm_bench.telemetry.discovery import cpu_name, discover_gpus
from llm_bench.telemetry.sampling import Sampler
from llm_bench.telemetry.schemas import Observation


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def fake_sys(root: Path) -> Path:
    device = root / "devices/pci/0000:01:00.0"
    write(device / "vendor", "0x1002")
    write(device / "device", "0x1586")
    write(device / "product_name", "Synthetic Radeon")
    write(device / "gpu_busy_percent", "75")
    write(device / "mem_info_vram_used", "123456789")
    write(device / "mem_info_gtt_used", "987654321")
    for name in ("card0", "card1"):
        card = root / "class/drm" / name
        card.mkdir(parents=True)
        (card / "device").symlink_to(device, target_is_directory=True)
    hwmon = root / "class/hwmon/hwmon0"
    write(hwmon / "name", "amdgpu")
    (hwmon / "device").symlink_to(device, target_is_directory=True)
    for name, value in {
        "temp1_input": "65000",
        "power1_average": "40000000",
        "freq1_input": "1500000000",
        "freq2_input": "800000000",
        "fan1_input": "1400",
    }.items():
        write(hwmon / name, value)
    return device


def test_discovery_and_amd_sensor_units(tmp_path: Path) -> None:
    fake_sys(tmp_path)
    write(tmp_path / "proc/cpuinfo", "processor : 0\nmodel name : Synthetic Ryzen\n")
    assert cpu_name(tmp_path / "proc") == "Synthetic Ryzen"
    assert len(discover_gpus(tmp_path)) == 1
    collector = Collector(tmp_path)
    rows = {
        cap.metric: (cap, value)
        for cap, value, _ in collector.read()
        if cap.scope in {"gpu", "package_or_soc", "amdgpu"}
    }
    assert rows["gpu_utilization"][1] == 75
    assert rows["vram_used"][1] == 123456789
    assert rows["gtt_used"][1] == 987654321
    assert rows["power"][1] == 40
    assert rows["power"][0].scope == "package_or_soc"
    assert rows["temperature"][1] == 65
    assert rows["gpu_frequency"][1] == 1_500_000_000
    assert rows["gpu_memory_frequency"][1] == 800_000_000
    assert rows["fan_speed"][1] == 1400


def test_missing_and_permission_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector = Collector(tmp_path)
    assert any(c.metric == "power" and not c.available for c in collector.capabilities)

    def denied() -> float:
        raise PermissionError("fixture")

    collector.add("power", "denied", denied)
    collector.add("power", "malformed", lambda: float("nan"))
    collector.add("gpu_utilization", "out-of-range", lambda: 101)
    rows = {cap.source: (value, reason) for cap, value, reason in collector.read()}
    assert rows["denied"] == (None, "PermissionError")
    assert rows["malformed"] == (None, "ValueError")
    assert rows["out-of-range"] == (None, "ValueError")
    monkeypatch.setattr("llm_bench.telemetry.discovery.platform.processor", lambda: "")
    assert cpu_name(tmp_path / "missing") != "unknown"


def test_smi_json_does_not_guess_device_order_or_missing_values() -> None:
    values = parse_smi(
        {
            "card0": {
                "PCI Bus": "0000:01:00.0",
                "GPU use (%)": "72",
                "VRAM Total Used Memory (B)": "N/A",
            }
        },
        "rocm-smi",
    )
    assert values == {"0000:01:00.0": {"gpu_utilization": 72}}
    assert parse_smi({"card0": {"GPU use (%)": "99"}}, "rocm-smi") == {}
    assert parse_smi(
        {"gpu_data": [{"bdf": "0000:01:00.0", "usage": {"gfx_activity": 55}}]}, "amd-smi"
    ) == {"0000:01:00.0": {"gpu_utilization": 55}}
    assert parse_smi("bad", "amd-smi") == {}


def test_smi_timeout_disables_repeated_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_sys(tmp_path)
    collector = Collector(tmp_path)
    collector.smi = "/fake/rocm-smi"
    calls = []

    def timeout(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise subprocess.TimeoutExpired("rocm-smi", 0.5)

    monkeypatch.setattr(subprocess, "run", timeout)
    collector.read()
    collector.read()
    assert len(calls) == 1
    assert calls[0]["timeout"] == 0.5
    assert collector.smi_disabled == "TimeoutExpired"


class FakeCollector:
    def __init__(self) -> None:
        self.capabilities = [capability("power", "fixture", scope="package")]
        self.gpus: list[dict[str, str]] = []
        self.collected = threading.Event()
        self.reads = 0

    def prime(self) -> None:
        pass

    def read(self) -> list[tuple[Any, Any, Any]]:
        self.reads += 1
        if self.reads >= 3:
            self.collected.set()
        return [(self.capabilities[0], 40, None)]


def test_sampling_lifecycle_periodic_boundaries_and_failure() -> None:
    collector = FakeCollector()
    with Sampler(collector, 0.005) as sampler:
        with sampler.invocation("one") as first:
            assert collector.collected.wait(2)
        with pytest.raises(RuntimeError), sampler.invocation("two") as second:
            raise RuntimeError("inference failed")
    assert sampler.thread is not None and not sampler.thread.is_alive()
    assert len({o.sequence for o in first.observations}) >= 3
    assert {o.invocation_id for o in first.observations} == {"one"}
    assert {o.invocation_id for o in second.observations} == {"two"}
    assert first.observations[0].offset_seconds <= 0
    assert first.observations[-1].offset_seconds >= first.end - first.start
    assert sampler.stats.collection_count >= 5


def observation(t: float, value: float | None, metric: str = "power", **kwargs: Any) -> Observation:
    return Observation(
        **capability(metric, "fixture", **kwargs).model_dump(),
        invocation_id="run",
        sequence=int((t + 1) * 10),
        timestamp_utc=datetime.now(timezone.utc),
        offset_seconds=t,
        value_float=value,
    )


@pytest.mark.parametrize(
    "metric,values,expected",
    [
        ("power", [40, 40, 40], 40),
        ("power", [0, 40, 80], 40),
        ("energy", [0, 20, 40], 40),
    ],
)
def test_energy_known_traces(metric: str, values: list[float], expected: float) -> None:
    rows = [observation(t, value, metric) for t, value in zip([0, 0.5, 1], values, strict=True)]
    summary = summarize_telemetry(rows, 1, 0.5, 10)[0]
    assert summary.energy_joules == pytest.approx(expected)
    assert summary.joules_per_generated_token == pytest.approx(expected / 10)
    assert summary.minimum == min(values)
    assert summary.maximum == max(values)
    assert summary.coverage_fraction == 1


def test_energy_gaps_counter_resets_and_boundary_clipping() -> None:
    assert (
        summarize_telemetry([observation(0, 40), observation(1, 40)], 1, 0.1, 1)[0].energy_joules
        is None
    )
    assert (
        summarize_telemetry(
            [observation(0, 40), observation(0.5, None), observation(1, 40)], 1, 1, 1
        )[0].energy_joules
        is None
    )
    assert (
        summarize_telemetry([observation(0, 90, "energy"), observation(1, 10, "energy")], 1, 1, 1)[
            0
        ].energy_joules
        is None
    )
    wrapped = [
        observation(0, 90, "energy", counter_range=100),
        observation(1, 10, "energy", counter_range=100),
    ]
    assert summarize_telemetry(wrapped, 1, 1, 0)[0].energy_joules == 20
    assert summarize_telemetry(wrapped, 1, 1, 0)[0].joules_per_generated_token is None
    clipped = summarize_telemetry([observation(-0.1, 40), observation(1.1, 40)], 1, 1, 10)[0]
    assert clipped.energy_joules == pytest.approx(40)
    assert clipped.coverage_seconds == pytest.approx(1)


def test_runner_all_phases_associated_and_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_bench.runner.Collector", FakeCollector)
    result = run_benchmark(BenchmarkConfig(model="qwen", repeat=1, warmup=1), FakeBackend())
    ids = {s.invocation_id for s in result.samples}
    assert len(ids) == len(result.samples)
    assert {o.invocation_id for o in result.telemetry} == ids
    assert {s.phase for s in result.samples} == {"cold_start", "warmup", "warm_start"}
    assert result.schema_version == "1.2"
    disabled = run_benchmark(
        BenchmarkConfig(model="qwen", repeat=1, warmup=0, telemetry_enabled=False), FakeBackend()
    )
    assert not disabled.telemetry

    def unavailable() -> None:
        raise PermissionError("no hardware")

    monkeypatch.setattr("llm_bench.runner.Collector", unavailable)
    unavailable_result = run_benchmark(
        BenchmarkConfig(model="qwen", repeat=1, warmup=0), FakeBackend()
    )
    assert unavailable_result.samples and not unavailable_result.telemetry
    assert any("discovery unavailable" in w for w in unavailable_result.warnings)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(model="qwen", telemetry_interval_seconds=interval)


def test_process_identity_descendants_and_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    class Process:
        def __init__(self, pid: int, name: str) -> None:
            self.pid = pid
            self.info = {"name": name, "ppid": 1 if pid == 2 else 0}

        def name(self):
            return self.info["name"]

        def create_time(self):
            return 123.0

        def children(self, recursive=False):
            return [child] if self.pid == 1 else []

        def cpu_percent(self, interval):
            return 150.0

        def memory_info(self):
            return SimpleNamespace(rss=123456)

        def num_threads(self):
            return 4

    parent, child = Process(1, "ollama"), Process(2, "llama-server")
    monkeypatch.setattr("psutil.process_iter", lambda attrs: [parent, child])
    collector = Collector(tmp_path)
    first = collector._read_processes()
    assert {c.device for c, _, _ in first} == {"pid:1:123.0", "pid:2:123.0"}
    assert all(v is None for c, v, _ in first if c.metric == "process_cpu")
    second = collector._read_processes()
    assert all(v == 150 for c, v, _ in second if c.metric == "process_cpu")
    monkeypatch.setattr("psutil.process_iter", lambda attrs: [])
    missing = collector._read_processes()
    assert not collector.processes
    assert all(v is None and reason for _, v, reason in missing)


def test_collector_failures_cleanup_and_missed_deadlines() -> None:
    class Broken(FakeCollector):
        def read(self):
            raise PermissionError("fixture")

    with Sampler(Broken(), 0.001) as broken:
        with broken.invocation("broken"):
            pass
    assert broken.stats.errors and not broken.thread.is_alive()

    class Slow(FakeCollector):
        def read(self):
            threading.Event().wait(0.005)
            return super().read()

    slow_collector = Slow()
    with Sampler(slow_collector, 0.001) as slow:
        with slow.invocation("slow"):
            assert slow_collector.collected.wait(2)
    assert slow.stats.missed_deadlines > 0


def test_powercap_symlink_discovery_and_unreadable_host_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_bench.hardware import collect_hardware

    zone = tmp_path / "devices/virtual/powercap/intel-rapl:0"
    write(zone / "energy_uj", "1500000")
    write(zone / "max_energy_range_uj", "10000000")
    write(zone / "name", "package-0")
    classes = tmp_path / "class/powercap"
    classes.mkdir(parents=True)
    (classes / "intel-rapl:0").symlink_to(zone, target_is_directory=True)
    collector = Collector(tmp_path)
    energy = [(cap, value) for cap, value, reason in collector.read() if cap.metric == "energy"]
    assert energy[0][1] == 1.5
    assert energy[0][0].counter_range == 10

    def denied():
        raise PermissionError("fixture")

    monkeypatch.setattr("psutil.virtual_memory", denied)
    assert collect_hardware().total_ram_bytes is None
