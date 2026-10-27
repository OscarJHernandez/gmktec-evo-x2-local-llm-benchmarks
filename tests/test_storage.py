"""Locking and cleanup safety: only recognised, owned files are ever removed."""

from __future__ import annotations

import contextvars
import json
import os
from pathlib import Path
from typing import Any

import pytest
import typer
from factories import fixture_experiment

from llm_bench import storage
from llm_bench.analytical import export_snapshot
from llm_bench.persistence import save_experiment
from llm_bench.storage import (
    clean_results,
    cleanup_candidates,
    locked_command,
    output_lock,
)

REPORT = "<html><title>Local LLM benchmark report</title></html>"


def in_fresh_context(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run without the re-entrancy shortcut so a second lock really contends."""
    return contextvars.Context().run(function, *args, **kwargs)


@pytest.fixture(autouse=True)
def private_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    locks = tmp_path / "tmp"
    locks.mkdir()
    monkeypatch.setattr(storage.tempfile, "gettempdir", lambda: str(locks))
    return locks


def test_lock_is_reentrant_exclusive_and_released(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()

    def acquire() -> None:
        with output_lock(root):
            pass

    with output_lock(root):
        with output_lock(root):
            pass
        with pytest.raises(ValueError, match="busy"):
            in_fresh_context(acquire)
    in_fresh_context(acquire)


def test_lock_rejects_symlinked_lock_directory(tmp_path: Path, private_lock_dir: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (private_lock_dir / f"llm-bench-locks-{os.getuid()}").symlink_to(
        elsewhere, target_is_directory=True
    )
    with pytest.raises(ValueError, match="Untrusted"):
        with output_lock(tmp_path):
            pass


def test_locked_command_holds_lock_and_exits_cleanly_when_busy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    seen: list[str] = []

    @locked_command
    def command(output_dir: Path, other: int = 0) -> str:
        def contend() -> None:
            with output_lock(output_dir):
                pass

        with pytest.raises(ValueError, match="busy"):
            in_fresh_context(contend)
        seen.append("ran")
        return f"done {other}"

    assert command(root, other=2) == "done 2"
    assert seen == ["ran"]
    with output_lock(root):
        with pytest.raises(typer.Exit) as raised:
            in_fresh_context(command, root)
    assert raised.value.exit_code == 1
    assert "busy" in capsys.readouterr().err


def test_locked_command_locks_parent_of_file_arguments(tmp_path: Path) -> None:
    @locked_command
    def command(output: Path) -> None:
        raise AssertionError("must not run while the directory is locked")

    with output_lock(tmp_path):
        with pytest.raises(typer.Exit):
            in_fresh_context(command, tmp_path / "report.html")


def test_cleanup_only_removes_recognised_files(tmp_path: Path) -> None:
    root = tmp_path / "results"
    save_experiment(fixture_experiment(False), root / "model.json")
    (root / "model.csv").write_text("model,workload,phase,repetition\nm,w,p,1\n")
    (root / "other.json").write_text("{}")
    (root / "unrelated.csv").write_text("a,b\n1,2\n")
    (root / "benchmark.log").write_text("log")
    (root / "report.html").write_text(REPORT)
    (root / "mine.html").write_text("<html>someone else's page</html>")
    (root / "catalog-progress-1.jsonl").write_text(
        json.dumps({"model": "m", "status": "done", "candidate_index": 0}) + "\n"
    )
    (root / "catalog-progress-2.jsonl").write_text('{"model": "m"}\n')
    (root / "catalog-progress-3.jsonl").write_text("{broken\n")
    (root / "benchmark-state-1.jsonl").write_text("not a progress state")
    names = {path.name for path in cleanup_candidates(root)}
    assert names == {"model.json", "model.csv", "benchmark.log", "report.html"} | {
        "catalog-progress-1.jsonl"
    }


def test_cleanup_ignores_symlinks_and_csv_with_wrong_header(tmp_path: Path) -> None:
    root = tmp_path / "results"
    save_experiment(fixture_experiment(False), root / "model.json")
    (root / "model.csv").write_text("a,b\n1,2\n")
    outside = tmp_path / "outside.log"
    outside.write_text("keep")
    (root / "benchmark.log").symlink_to(outside)
    assert {p.name for p in cleanup_candidates(root)} == {"model.json"}


def test_cleanup_handles_legacy_exports_and_rejects_foreign_manifests(tmp_path: Path) -> None:
    root = tmp_path / "results"
    legacy = root / "exports"
    legacy.mkdir(parents=True)
    (legacy / "observations-1.csv").write_text("model,workload,phase,repetition\n")
    (legacy / "observations-2.csv").write_text("x\n")
    (legacy / "observations-3.jsonl").write_text(
        json.dumps({"experiment_id": "e", "metric": "m", "value": 1}) + "\n"
    )
    (legacy / "report.html").write_text(REPORT)
    (legacy / "notes.txt").write_text("keep")

    experiment = fixture_experiment()
    export_snapshot([experiment], legacy / "good")
    foreign = export_snapshot([experiment], legacy / "foreign-file")
    wrong_format = export_snapshot([experiment], legacy / "wrong-format")
    linked = export_snapshot([experiment], legacy / "linked")

    manifest = json.loads((foreign / "manifest.json").read_text())
    manifest["files"].append("../../precious.txt")
    (foreign / "manifest.json").write_text(json.dumps(manifest))
    manifest = json.loads((wrong_format / "manifest.json").read_text())
    manifest["format"] = "something-else"
    (wrong_format / "manifest.json").write_text(json.dumps(manifest))
    (linked / "experiments.parquet").rename(tmp_path / "moved.parquet")
    (linked / "experiments.parquet").symlink_to(tmp_path / "moved.parquet")
    (legacy / "broken").mkdir()
    (legacy / "broken" / "manifest.json").write_text("{not json")

    names = {p.relative_to(legacy).as_posix() for p in cleanup_candidates(root)}
    assert {
        "observations-1.csv",
        "observations-3.jsonl",
        "report.html",
        "good/manifest.json",
        "good/experiments.parquet",
    } <= names
    assert "observations-2.csv" not in names and "notes.txt" not in names
    assert not any(
        name.startswith(("foreign-file/", "wrong-format/", "linked/", "broken/")) for name in names
    )


def test_clean_removes_manifest_last_and_prunes_empty_snapshot_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    snapshot = export_snapshot([fixture_experiment()], root / "exports" / "one")
    (snapshot / "extra.txt").write_text("unknown file keeps the directory")
    removed, preserved = clean_results(root)
    assert snapshot / "manifest.json" in removed
    assert snapshot.is_dir() and (snapshot / "extra.txt").exists()
    assert root / "exports" in preserved

    (snapshot / "extra.txt").unlink()
    other = export_snapshot([fixture_experiment()], root / "exports" / "two")
    clean_results(root)
    assert not other.exists()


def test_clean_dry_run_deletes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "results"
    save_experiment(fixture_experiment(False), root / "model.json")
    files, _ = clean_results(root, dry_run=True)
    assert files == [root / "model.json"] and (root / "model.json").exists()
    assert clean_results(tmp_path / "missing") == ([], [])


def test_clean_aborts_if_candidate_becomes_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"
    root.mkdir()
    target = tmp_path / "precious.txt"
    target.write_text("keep")
    link = root / "model.json"
    link.symlink_to(target)
    monkeypatch.setattr(storage, "cleanup_candidates", lambda _: [link])
    with pytest.raises(ValueError, match="became a symlink"):
        clean_results(root)
    assert target.read_text() == "keep"
