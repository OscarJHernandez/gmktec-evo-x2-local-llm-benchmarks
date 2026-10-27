"""CLI command tests with every external dependency (Ollama, sensors, network) replaced."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from factories import FakeBackend, make_experiment
from typer.testing import CliRunner

from llm_bench import cli
from llm_bench.backends.ollama import OllamaError
from llm_bench.telemetry.collectors import capability

runner = CliRunner()


class DownBackend:
    def health(self) -> str:
        raise OllamaError("Cannot communicate with configured Ollama endpoint.")

    def models(self) -> list[dict[str, Any]]:
        raise OllamaError("Cannot communicate with configured Ollama endpoint.")


class StubCollector:
    smi = None
    capabilities = [
        capability("power", "fixture"),
        capability("temperature", "fixture", available=False, reason="no sensor"),
    ]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic doctor: fixed hardware, stub sensors, fake Ollama."""
    import llm_bench.hardware
    import llm_bench.telemetry.collectors as collectors

    monkeypatch.setattr(llm_bench.hardware, "collect_hardware", lambda: make_experiment().hardware)
    monkeypatch.setattr(collectors, "Collector", StubCollector)
    monkeypatch.setattr(cli, "_backend", lambda url: FakeBackend())


def test_doctor_reports_environment_and_capabilities(offline: None) -> None:
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "no sensor" in result.output
    assert "not detected" in result.output
    assert "Ollama" in result.output and "test" in result.output


def test_doctor_fails_when_ollama_is_down(offline: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_backend", lambda url: DownBackend())
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "Ollama unavailable" in result.output


def test_doctor_survives_sensor_discovery_failure(
    offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llm_bench.telemetry.collectors as collectors

    def broken() -> None:
        raise OSError("sysfs unreadable")

    monkeypatch.setattr(collectors, "Collector", broken)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "Telemetry discovery unavailable: OSError" in result.output


def test_models_lists_installed_and_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_backend", lambda url: FakeBackend())
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0 and "qwen:latest" in result.output
    monkeypatch.setattr(cli, "_backend", lambda url: DownBackend())
    failed = runner.invoke(cli.app, ["models"])
    assert failed.exit_code == 1 and "Cannot communicate" in failed.output


def test_physics_prepare_validates_dataset_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(cli.app, ["physics-prepare", "--dataset", "nope"]).exit_code == 2
    args = ["physics-prepare", "--dataset", "scibench", "--data-dir", str(tmp_path)]

    def failing(dataset: str, destination: Path) -> list[Path]:
        raise RuntimeError("offline")

    monkeypatch.setattr(cli, "prepare_physics", failing)
    failed = runner.invoke(cli.app, args)
    assert failed.exit_code == 1 and "download failed: offline" in failed.output

    monkeypatch.setattr(cli, "prepare_physics", lambda dataset, destination: [Path("a")])
    ok = runner.invoke(cli.app, args)
    assert ok.exit_code == 0 and "1 cached source file" in ok.output


@pytest.mark.parametrize(
    "args",
    [
        ["--dataset", "nope"],
        ["--models", " , "],
        ["--models", "installed,qwen"],
    ],
)
def test_physics_run_rejects_bad_selection(args: list[str]) -> None:
    result = runner.invoke(cli.app, ["physics-run", *args])
    assert result.exit_code == 2


def test_physics_run_fetches_runs_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli, "prepare_physics", lambda dataset, root: calls.setdefault("fetch", 1))

    def fake_run(**kwargs: Any) -> dict[str, int]:
        calls.update(kwargs)
        return {"attempted": 2}

    monkeypatch.setattr(cli, "run_physics", fake_run)
    args = ["physics-run", "--fetch", "--models", "a, b", "--output-dir", str(tmp_path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls["fetch"] == 1 and calls["models"] == ["a", "b"]
    assert calls["output"] == tmp_path / "scibench-attempts.jsonl"
    assert "not official benchmark scores" in result.output

    def failing_run(**kwargs: Any) -> dict[str, int]:
        raise ValueError("Missing installed Ollama model(s): a")

    monkeypatch.setattr(cli, "run_physics", failing_run)
    failed = runner.invoke(cli.app, args)
    assert failed.exit_code == 1 and "Physics benchmark failed" in failed.output


def test_physics_results_table_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.invoke(cli.app, ["physics-results", "--dataset", "nope"]).exit_code == 2
    missing = runner.invoke(cli.app, ["physics-results", "--output-dir", str(tmp_path)])
    assert missing.exit_code == 1 and "Could not read physics results" in missing.output

    row = {
        "attempts": 4,
        "scored": 3,
        "correct": 2,
        "unscored": 1,
        "inference_errors": 0,
        "accuracy_numeric_proxy": 2 / 3,
    }
    empty = dict(row, scored=0, correct=0, accuracy_numeric_proxy=None)
    monkeypatch.setattr(
        cli, "physics_summary", lambda path: {("m1", "cfg-a"): row, ("m2", "cfg-b"): empty}
    )
    result = runner.invoke(cli.app, ["physics-results", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "66.67%" in result.output and "n/a" in result.output
