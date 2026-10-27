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

pytestmark = pytest.mark.usefixtures("benchmark_machine")


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

        def pull(self, name: str, **kwargs: Any) -> None:
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
            "--discovery",
            "curated",
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
        "failed",
        "skipped_fit",
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
        [
            "catalog-run",
            "--discovery",
            "curated",
            "--model-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    logs = list(tmp_path.glob("catalog-progress-*.jsonl"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text().splitlines()[0])["status"] == "skipped_not_installed"


def test_live_catalog_end_to_end_progress_fit_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from factories import FakeBackend

    from llm_bench.discovery import Discovery
    from llm_bench.progress import latest_progress, status_text

    class Backend(FakeBackend):
        def models(self) -> list[dict[str, Any]]:
            return []

        def pull(self, name: str, **kwargs: Any) -> None:
            assert "downloading" in status_text(tmp_path)
            kwargs["progress"]({"status": "pulling layer", "completed": 50, "total": 100})
            assert "50%" in status_text(tmp_path)
            if name == "broken:1b":
                raise OllamaError("unavailable")

        def find_model(self, name: str) -> dict[str, Any]:
            return {"name": name, "size": 1_000_000_000}

    backend = Backend()
    monkeypatch.setattr(cli, "_backend", lambda _: backend)
    monkeypatch.setattr(cli, "system_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(cli, "free_storage_bytes", lambda _: 1000 * 1024**3)
    monkeypatch.setattr(
        cli,
        "discover_library",
        lambda **_: Discovery(
            candidates=[
                Candidate("ok:1b", 1),
                Candidate("broken:1b", 1),
                Candidate("huge:999b", 900),
            ],
            warnings=["unavailable family"],
            families=4,
        ),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "catalog-run",
            "--auto-pull",
            "--no-telemetry",
            "--repeat",
            "2",
            "--warmup",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    state = latest_progress(tmp_path)
    assert state is not None
    assert (state.completed, state.failed, state.skipped) == (1, 1, 1)
    assert state.finished and state.discovery_warnings == ["unavailable family"]
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert "Remaining: 0" in status_text(tmp_path)
    assert backend.calls == 10


def test_catalog_preview_is_read_only_and_reuses_installed_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_bench.discovery import Discovery

    class Backend:
        def models(self) -> list[dict[str, Any]]:
            return [{"name": "test:latest", "size": 1_000_000_000, "digest": "abc"}]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    monkeypatch.setattr(
        cli, "discover_library", lambda **_: Discovery(candidates=[Candidate("test:1b", 1, "abc")])
    )
    result = CliRunner().invoke(
        cli.app,
        ["catalog-run", "--auto-pull", "--dry-run", "--output-dir", str(tmp_path / "preview")],
    )
    assert result.exit_code == 0, result.output
    assert "1 candidates" in result.output and "benchmark installed" in result.output
    assert not (tmp_path / "preview").exists()


def test_catalog_rechecks_space_and_post_pull_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_bench.discovery import Discovery
    from llm_bench.progress import latest_progress

    pulled = []

    class Backend:
        def models(self) -> list[dict[str, Any]]:
            return []

        def pull(self, name: str, **kwargs: Any) -> None:
            pulled.append(name)

        def find_model(self, name: str) -> dict[str, Any]:
            # Mutable registry tag now much larger than discovery metadata.
            return {"name": name, "size": 300_000_000_000}

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    monkeypatch.setattr(cli, "system_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(cli, "free_storage_bytes", lambda _: 1024**3 if pulled else 100 * 1024**3)
    monkeypatch.setattr(
        cli,
        "discover_library",
        lambda **_: Discovery(candidates=[Candidate("test:1b", 1), Candidate("test:2b", 2)]),
    )
    result = CliRunner().invoke(
        cli.app, ["catalog-run", "--auto-pull", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert pulled == ["test:1b"]
    state = latest_progress(tmp_path)
    assert state is not None and state.skipped == 2 and state.completed == 0
    assert "storage" in str(state.detail)
    assert not list(tmp_path.glob("*.json"))


def test_installed_alias_deduplication_and_safe_community_names() -> None:
    from llm_bench.model_catalog import unique_candidates
    from llm_bench.models import BenchmarkConfig

    installed = select_installed(
        [
            {"name": "user/model:8b", "size": 1000000000, "digest": "sha256:abc"},
            {"name": "user/model:latest", "size": 1000000000, "digest": "abc"},
            {"name": "malformed:1b", "size": 1000000000, "details": "broken"},
        ]
    )
    assert [item.name for item in unique_candidates(installed)] == ["user/model:8b"]
    assert BenchmarkConfig(model="user/model:8b").model == "user/model:8b"
    for bad_name in ["../model", "/model", "user/../model", "https://host/model", "model:../x"]:
        with pytest.raises(ValueError):
            BenchmarkConfig(model=bad_name)


def test_installed_only_never_discovers_or_pulls_and_reuses_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from factories import FakeBackend

    from llm_bench.progress import latest_progress

    class Backend(FakeBackend):
        def models(self) -> list[dict[str, Any]]:
            return [
                {"name": tag, "size": 1000000000, "digest": "abc"}
                for tag in ["user/model:8b", "user/model:latest"]
            ]

        def find_model(self, name: str) -> dict[str, Any]:
            return self.models()[0]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())

    def unexpected(**kwargs: Any) -> None:
        pytest.fail("Installed-only runs must not discover or pull")

    monkeypatch.setattr(cli, "discover_library", unexpected)
    result = CliRunner().invoke(
        cli.app,
        [
            "catalog-run",
            "--installed-only",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--no-telemetry",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    state = latest_progress(tmp_path)
    assert state is not None and state.completed == 1 and len(state.models) == 1


def test_catalog_reuses_installed_manifest_without_pulling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from factories import FakeBackend

    from llm_bench.discovery import Discovery

    class Backend(FakeBackend):
        def models(self) -> list[dict[str, Any]]:
            return [{"name": "test:latest", "size": 1000000000, "digest": "sha256:abc"}]

        def find_model(self, name: str) -> dict[str, Any]:
            assert name == "test:latest"
            return self.models()[0]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    monkeypatch.setattr(
        cli, "discover_library", lambda **_: Discovery(candidates=[Candidate("test:1b", 1, "abc")])
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "catalog-run",
            "--auto-pull",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--no-telemetry",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_named_model_preview_matches_make_selection_without_writing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Backend:
        def models(self) -> list[dict[str, Any]]:
            return [{"name": "test:latest"}]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    result = CliRunner().invoke(
        cli.app,
        ["compare", "--models", "test", "--dry-run", "--output-dir", str(tmp_path / "preview")],
    )
    assert result.exit_code == 0, result.output
    assert "Selected installed model: test" in result.output
    assert not (tmp_path / "preview").exists()


def test_curated_catalog_uses_actual_installed_size_and_deduplicates_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from factories import FakeBackend

    from llm_bench.progress import latest_progress

    class Backend(FakeBackend):
        def models(self) -> list[dict[str, Any]]:
            return [
                {"name": name, "size": 1000000000, "digest": "abc"}
                for name in ["test:1b", "test:latest"]
            ]

        def find_model(self, name: str) -> dict[str, Any]:
            return self.models()[0]

    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    monkeypatch.setattr(cli, "catalog_candidates", lambda _: [Candidate("test:1b", 900)])
    result = CliRunner().invoke(
        cli.app,
        [
            "catalog-run",
            "--discovery",
            "curated",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--no-telemetry",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    state = latest_progress(tmp_path)
    assert state is not None and state.completed == 1 and len(state.models) == 1
    assert state.models[0].size_gb == 1


@pytest.mark.parametrize(
    ("ram_gib", "disk_gib", "installed", "expected_status", "reason"),
    [
        (8, 100, False, "skipped_fit", "memory"),
        (128, 1, False, "skipped_fit", "storage"),
        (8, 100, True, "skipped_fit", "memory"),
        (128, 1, True, "completed", None),
    ],
)
def test_catalog_enforces_resource_limits_independently_of_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ram_gib: int,
    disk_gib: int,
    installed: bool,
    expected_status: str,
    reason: str | None,
) -> None:
    from llm_bench.progress import latest_progress

    class Backend:
        def models(self) -> list[dict[str, Any]]:
            return [{"name": "test:1b", "size": 1000000000}] if installed else []

        def pull(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("Resource-limited candidates must not be downloaded")

    completed: list[str] = []
    monkeypatch.setattr(cli, "_backend", lambda _: Backend())
    monkeypatch.setattr(cli, "catalog_candidates", lambda _: [Candidate("test:1b", 1)])
    monkeypatch.setattr(cli, "system_ram_bytes", lambda: ram_gib * 1024**3)
    monkeypatch.setattr(cli, "free_storage_bytes", lambda _: disk_gib * 1024**3)
    monkeypatch.setattr(cli, "_run", lambda model, *args, **kwargs: completed.append(model))
    result = CliRunner().invoke(
        cli.app,
        ["catalog-run", "--auto-pull", "--discovery", "curated", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == (0 if expected_status == "completed" else 1), result.output
    progress_path = next(tmp_path.glob("catalog-progress-*.jsonl"))
    record = json.loads(progress_path.read_text())
    assert record["status"] == expected_status
    assert (reason in record["reason"]) if reason is not None else record["reason"] is None
    assert completed == (["test:1b"] if expected_status == "completed" else [])
    state = latest_progress(tmp_path)
    assert state is not None
    assert state.completed == len(completed)
    assert state.skipped == int(expected_status == "skipped_fit")
