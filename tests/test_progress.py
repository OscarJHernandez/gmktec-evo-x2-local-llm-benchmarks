"""Progress association, interrupted writers, cleanup, and read-only status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_core import FakeBackend
from typer.testing import CliRunner

from llm_bench import cli
from llm_bench.model_catalog import Candidate
from llm_bench.models import BenchmarkConfig
from llm_bench.progress import Progress, latest_progress, status_text
from llm_bench.runner import run_benchmark
from llm_bench.storage import clean_results, output_lock

pytestmark = pytest.mark.usefixtures("benchmark_machine")


def test_progress_counts_active_download_and_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = Progress(tmp_path, 50, 3)
    tracker.plan([Candidate(f"test:{i}b", 1.0) for i in range(4)])
    tracker.start_model("test:0b")
    tracker.terminal("completed")
    tracker.start_model("test:1b")
    tracker.terminal("failed", "bad response")
    tracker.start_model("test:2b")
    tracker.terminal("skipped_fit", "too large")
    tracker.start_model("test:3b")
    tracker.download({"completed": 20, "total": 100, "status": "pulling layer"})
    text = status_text(tmp_path)
    assert "RUNNING" in text and "75%" in text and "20%" in text
    assert "Benchmarked: 1 | Failed: 1 | Skipped: 1" in text
    assert "Remaining: 1" in text
    monkeypatch.setattr("llm_bench.progress.process_alive", lambda _: False)
    assert "STOPPED / INTERRUPTED" in status_text(tmp_path)
    tracker.invocation(2, 160, "long-input", "warmup", 2)
    assert "2/160" in status_text(tmp_path) and "warmup" in status_text(tmp_path)


def test_progress_read_during_writer_lock_and_safe_cleanup(tmp_path: Path) -> None:
    outside = tmp_path / "keep.jsonl"
    outside.write_text("keep me")
    with output_lock(tmp_path), Progress(tmp_path, 2, 0) as tracker:
        tracker.plan([Candidate("test:1b", 1.0)])
        tracker.start_model("test:1b")
        tracker.terminal("completed")
        response = CliRunner().invoke(cli.app, ["status", "--output-dir", str(tmp_path)])
        assert response.exit_code == 0 and "Benchmarked: 1" in response.output
    symlink = tmp_path / "benchmark-state-zzz.jsonl"
    symlink.symlink_to(outside)
    corrupt = tmp_path / "benchmark-state-zz.jsonl"
    corrupt.write_text("{broken")
    assert "FINISHED" in status_text(tmp_path)
    files, _ = clean_results(tmp_path, dry_run=True)
    assert files == [tracker.path] and tracker.path.exists()
    clean_results(tmp_path)
    assert outside.exists() and symlink.is_symlink() and corrupt.exists()
    assert "No benchmark progress" in status_text(tmp_path)


def test_new_sweep_resets_counts_and_uses_unique_ids(tmp_path: Path) -> None:
    with Progress(tmp_path, 50, 3) as first:
        first.terminal("completed")
    with Progress(tmp_path, 50, 3) as second:
        state = latest_progress(tmp_path)
        assert state is not None and state.run_id != first.state.run_id
        assert state.completed == 0 and state.run_id == second.state.run_id


def test_runner_reports_each_invocation_and_uses_fifty_measured_samples() -> None:
    events: list[tuple[Any, ...]] = []
    experiment = run_benchmark(
        BenchmarkConfig(model="test", telemetry_enabled=False),
        FakeBackend(),
        progress=lambda *args: events.append(args),
    )
    assert len(experiment.samples) == 160
    assert sum(s.phase == "warm_start" for s in experiment.samples) == 150
    assert events[0][:2] == (0, 160) and events[-1][:2] == (160, 160)
    assert len(events) == 320
    assert len([s for s in experiment.samples if s.phase == "warmup"]) == 9
    assert BenchmarkConfig(model="test", repeat=1000).repeat == 1000
    with pytest.raises(ValueError):
        BenchmarkConfig(model="test", repeat=1001)


def test_download_throttling_resets_for_models_and_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llm_bench.progress.time.monotonic", lambda: 10.0)
    tracker = Progress(tmp_path, 1, 0)
    tracker.start_model("first:1b")
    tracker.download({"status": "pulling manifest"})
    tracker.download({"status": "pulling", "digest": "a", "completed": 1, "total": 10})
    assert tracker.state.download_done == 1
    tracker.download({"status": "pulling", "digest": "a", "completed": 2, "total": 10})
    assert tracker.state.download_done == 1  # Same-layer updates are rate limited.
    tracker.download({"status": "pulling", "digest": "b", "completed": 3, "total": 10})
    assert tracker.state.download_done == 3  # A new layer always appears immediately.
    tracker.start_model("second:1b")
    tracker.download({"status": "pulling", "digest": "b", "completed": 5, "total": 10})
    assert tracker.state.download_done == 5
    tracker.download({"status": "pulling", "digest": "b", "completed": 10, "total": 10})
    assert tracker.state.download_done == 10  # Completion is never hidden by throttling.


def test_process_identity_rejects_reused_pids_and_zombies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil

    from llm_bench.progress import process_alive

    tracker = Progress(tmp_path, 1, 0)
    assert process_alive(tracker.state)

    class Process:
        def create_time(self) -> float:
            return tracker.state.process_created + 1

        def status(self) -> str:
            return psutil.STATUS_RUNNING

    monkeypatch.setattr("llm_bench.progress.psutil.Process", lambda *_: Process())
    assert not process_alive(tracker.state)
    monkeypatch.setattr(Process, "create_time", lambda _: tracker.state.process_created)
    assert process_alive(tracker.state)
    monkeypatch.setattr(Process, "status", lambda _: psutil.STATUS_ZOMBIE)
    assert not process_alive(tracker.state)


def test_discovery_status_and_exception_outcomes(tmp_path: Path) -> None:
    tracker = Progress(tmp_path, 50, 3)
    tracker.update(stage="discovering", detail="Library families inspected: 3/10")
    text = status_text(tmp_path)
    assert "pending discovery" in text and "3/10" in text
    assert "Remaining: 0" not in text
    with pytest.raises(KeyboardInterrupt), Progress(tmp_path, 50, 3):
        raise KeyboardInterrupt
    assert "STOPPED / INTERRUPTED" in status_text(tmp_path)
    with pytest.raises(ValueError), Progress(tmp_path, 50, 3):
        raise ValueError("bad input")
    assert "FAILED" in status_text(tmp_path)


@pytest.mark.parametrize(
    "selection",
    [
        ["run", "--model", "test"],
        ["compare", "--models", "test"],
        ["catalog-run", "--installed-only"],
    ],
)
def test_inference_command_defaults_consistently_use_fifty_repeats(
    selection: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_bench.persistence import load_experiments

    class Backend(FakeBackend):
        def models(self) -> list[dict[str, Any]]:
            return [{"name": "test", "size": 1000000000}]

        def find_model(self, name: str) -> dict[str, Any]:
            return self.models()[0]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    result = CliRunner().invoke(
        cli.app, [*selection, "--no-telemetry", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    experiment = load_experiments([tmp_path])[0]
    assert experiment.repetition_count == 50 and experiment.warmup_count == 3
    state = latest_progress(tmp_path)
    assert state is not None and state.invocation_done == state.invocation_total == 160
    assert state.completed == 1
