"""Catalog fit, discovery, and isolated auto-pull execution tests (no real downloads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_bench import cli
from llm_bench.backends.ollama import OllamaError
from llm_bench.model_catalog import (
    CATALOG,
    Candidate,
    catalog_candidates,
    fit_reason,
    select_installed,
)


def test_catalog_covers_broad_local_models_without_duplicate_tags() -> None:
    names = [model.name for model in CATALOG]
    assert len(names) >= 40
    assert len(names) == len(set(names))
    assert "qwen3.5:35b" in names
    assert "qwen3-coder:30b" in names
    assert "gpt-oss:120b" in names
    assert "qwen3-coder:480b" not in names
    assert all("cloud" not in name and not name.endswith(":latest") for name in names)


def test_fit_has_memory_and_storage_headroom() -> None:
    small = Candidate("test:small", 5.0)
    large = Candidate("test:large", 120.0)
    ram = 128 * 1024**3
    assert fit_reason(small, total_ram_bytes=ram, free_disk_bytes=100 * 1024**3) is None
    assert "memory" in str(fit_reason(large, total_ram_bytes=ram, free_disk_bytes=500 * 1024**3))
    assert "storage" in str(fit_reason(small, total_ram_bytes=ram, free_disk_bytes=10 * 1024**3))
    assert fit_reason(small, total_ram_bytes=ram, free_disk_bytes=0, already_installed=True) is None


def test_catalog_extension_validates_sizes_and_filters_cloud(tmp_path: Path) -> None:
    additional = tmp_path / "catalog.json"
    additional.write_text('[{"name":"custom:8b","size_gb":5.5}]', encoding="utf-8")
    assert "custom:8b" in [item.name for item in catalog_candidates(additional)]
    additional.write_text('[{"name":"bad:1b","size_gb":-1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid model"):
        catalog_candidates(additional)
    additional.write_text('[{"name":"foo-cloud","size_gb":1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid model"):
        catalog_candidates(additional)


def test_installed_discovery_excludes_non_generating_and_cloud_models() -> None:
    models = [
        {"name": "custom:7b", "size": 4_500_000_000, "details": {}},
        {"name": "nomic-embed-text:latest", "size": 300_000_000, "details": {}},
        {"name": "remote-cloud:latest", "size": 0, "details": {}},
        {"name": "broken", "size": "unknown", "details": {}},
    ]
    assert select_installed(models) == [Candidate("custom:7b", 4.5)]


def test_catalog_auto_pulls_skips_failures_and_records_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Backend:
        def __init__(self) -> None:
            self.installed: dict[str, dict[str, Any]] = {}
            self.pulled: list[str] = []

        def models(self) -> list[dict[str, Any]]:
            return list(self.installed.values())

        def find_model(self, name: str) -> dict[str, Any]:
            return self.installed[name]

        def pull(self, name: str) -> None:
            self.pulled.append(name)
            if name == "unavailable:1b":
                raise OllamaError("Model not found")
            self.installed[name] = {"name": name, "size": 1_000_000_000}

    backend = Backend()
    completed: list[str] = []
    monkeypatch.setattr(cli, "_backend", lambda api_url: backend)
    monkeypatch.setattr(
        cli,
        "catalog_candidates",
        lambda extra: [
            Candidate("tiny:1b", 1.0),
            Candidate("huge:999b", 250.0),
            Candidate("unavailable:1b", 1.0),
        ],
    )
    monkeypatch.setattr(cli, "system_ram_bytes", lambda: 128 * 1024**3)

    def fake_run(name: str, *args: Any, **kwargs: Any) -> None:
        completed.append(name)

    monkeypatch.setattr(cli, "_run", fake_run)
    result = CliRunner().invoke(
        cli.app,
        [
            "catalog-run",
            "--auto-pull",
            "--model-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
            "--repeat",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert completed == ["tiny:1b"]
    assert backend.pulled == ["tiny:1b", "unavailable:1b"]
    logs = list(tmp_path.glob("catalog-progress-*.jsonl"))
    assert len(logs) == 1
    rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [row["status"] for row in rows] == [
        "completed",
        "skipped_fit",
        "failed",
    ]
    assert all(row["timestamp_utc"] for row in rows)


def test_catalog_defaults_not_to_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Backend:
        def models(self) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(cli, "_backend", lambda url: Backend())
    monkeypatch.setattr(cli, "catalog_candidates", lambda path: [Candidate("test:1b", 1.0)])
    result = CliRunner().invoke(
        cli.app,
        ["catalog-run", "--model-dir", str(tmp_path), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    logs = list(tmp_path.glob("catalog-progress-*.jsonl"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text().splitlines()[0])["status"] == "skipped_not_installed"
