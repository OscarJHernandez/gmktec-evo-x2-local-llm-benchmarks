from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from factories import fixture_experiment, make_experiment
from typer.testing import CliRunner

from llm_bench.analytical import SCHEMAS, export_snapshot
from llm_bench.cli import app
from llm_bench.persistence import load_experiments, save_experiment
from llm_bench.storage import clean_results, cleanup_candidates


def test_parquet_explicit_types_roundtrip_and_readers(tmp_path: Path) -> None:
    experiment = fixture_experiment()
    snapshot = export_snapshot([experiment, experiment], tmp_path / "snapshot")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["row_counts"]["experiments"] == 1
    for name, schema in SCHEMAS.items():
        table = pq.read_table(snapshot / f"{name}.parquet")
        assert table.schema == schema
        assert (
            pq.ParquetFile(snapshot / f"{name}.parquet").metadata.row_group(0).column(0).compression
            == "ZSTD"
        )
    assert pq.read_schema(snapshot / "telemetry.parquet").field(
        "timestamp_utc"
    ).type == pa.timestamp("us", tz="UTC")
    connection = duckdb.connect()
    runs = connection.read_parquet(str(snapshot / "inference_runs.parquet")).pl()
    assert isinstance(runs, pl.DataFrame)
    assert runs.height == 1
    assert pd.read_parquet(snapshot / "telemetry.parquet")["value_int"].notna().sum() == 3
    assert pl.read_parquet(snapshot / "telemetry_summary.parquet")["energy_joules"].drop_nulls()[
        0
    ] == pytest.approx(40)
    with pytest.raises(FileExistsError):
        export_snapshot([experiment], snapshot)
    with pytest.raises(ValueError, match="Conflicting"):
        export_snapshot(
            [experiment, experiment.model_copy(update={"model": "different"})],
            tmp_path / "conflict",
        )
    assert not (tmp_path / "conflict").exists()
    html = (snapshot / "report.html").read_text()
    assert "Hardware utilization and efficiency" in html
    assert "hardware-power" in html


def test_historical_json_stable_ids_empty_telemetry_and_atomic_save(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "examples/synthetic-example.json"
    one = load_experiments([source])[0]
    two = load_experiments([source])[0]
    assert one == two
    assert one.experiment_id == two.experiment_id
    assert one.samples[0].invocation_id == two.samples[0].invocation_id
    assert one.samples[0].started_at is None
    assert not one.telemetry
    snapshot = export_snapshot([one], tmp_path / "historical")
    assert pq.read_table(snapshot / "telemetry.parquet").num_rows == 0
    assert (
        pq.read_table(snapshot / "telemetry_summary.parquet").schema == SCHEMAS["telemetry_summary"]
    )
    output = save_experiment(one, tmp_path / "experiment.json")
    with pytest.raises(FileExistsError):
        save_experiment(two, output)
    assert not list(tmp_path.glob(".experiment-*"))


def test_export_rejects_orphan_and_duplicate_telemetry(tmp_path: Path) -> None:
    experiment = fixture_experiment()
    experiment.telemetry[0].invocation_id = "orphan"
    with pytest.raises(ValueError, match="Orphan"):
        export_snapshot([experiment], tmp_path / "orphan")
    experiment = fixture_experiment()
    experiment.telemetry.append(experiment.telemetry[0])
    with pytest.raises(ValueError, match="Duplicate"):
        export_snapshot([experiment], tmp_path / "duplicate")


def test_cleanup_preserves_unknown_physics_datasets_symlinks_and_fresh_ids(tmp_path: Path) -> None:
    root = tmp_path / "results"
    experiment = fixture_experiment()
    save_experiment(experiment, root / "model.json")
    export_snapshot([experiment], root / "exports/first")
    for filename in ("notes.txt", "unknown.json", "datasets/data.json", "physics/attempts.jsonl"):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserve me")
    outside = tmp_path / "outside.json"
    outside.write_text(experiment.model_dump_json())
    (root / "linked.json").symlink_to(outside)
    dry, _ = clean_results(root, dry_run=True)
    assert root / "model.json" in dry and (root / "model.json").exists()
    removed, preserved = clean_results(root)
    assert removed and root / "notes.txt" in preserved
    assert outside.exists() and (root / "linked.json").is_symlink()
    assert (root / "physics/attempts.jsonl").read_text() == "preserve me"
    assert not list(root.glob("model.json"))
    assert clean_results(root)[0] == []
    fresh = fixture_experiment(False)
    save_experiment(fresh, root / "fresh.json")
    # Export a selected experiment file when unrelated JSON shares its directory.
    snapshot = export_snapshot(load_experiments([root / "fresh.json"]), root / "exports/fresh")
    assert pq.read_table(snapshot / "experiments.parquet")["experiment_id"].to_pylist() == [
        fresh.experiment_id
    ]
    assert fresh.experiment_id != experiment.experiment_id
    assert not cleanup_candidates(tmp_path / "absent")
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        clean_results(tmp_path / "alias")


def test_clean_refuses_active_writer_and_reports_deletion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_bench.storage import output_lock

    root = tmp_path / "results"
    save_experiment(fixture_experiment(), root / "model.json")
    with output_lock(root):
        program = (
            "from pathlib import Path; from llm_bench.storage import clean_results; "
            "clean_results(Path(__import__('sys').argv[1]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(root)],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        assert result.returncode != 0 and "busy" in result.stderr
    original = Path.unlink

    def denied(path: Path, *args, **kwargs):
        if path.name == "model.json":
            raise PermissionError("fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    result = CliRunner().invoke(app, ["clean", "--output-dir", str(root)])
    assert result.exit_code == 1 and "Cleanup failed" in result.output


def test_cli_export_and_empty_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "results"
    save_experiment(fixture_experiment(False), root / "model.json")
    runner = CliRunner()
    result = runner.invoke(
        app, ["export", "--input", str(root), "--output-dir", str(root / "exports")]
    )
    assert result.exit_code == 0, result.output
    assert len(list((root / "exports").glob("*/manifest.json"))) == 1
    assert runner.invoke(app, ["clean", "--output-dir", str(tmp_path / "absent")]).exit_code == 0


def test_documented_sql_examples_execute(tmp_path: Path) -> None:
    import re

    snapshot = export_snapshot([fixture_experiment()], tmp_path / "snapshot")
    connection = duckdb.connect()
    for name in SCHEMAS:
        connection.read_parquet(str(snapshot / f"{name}.parquet")).create_view(name)
    document = (Path(__file__).parents[1] / "docs/analysis.md").read_text()
    queries = re.findall(r"```sql\n(.*?)```", document, flags=re.S)
    assert len(queries) == 3
    for query in queries:
        frame = connection.sql(query).pl()
        assert isinstance(frame, pl.DataFrame)
    connection.close()


def test_full_cli_fresh_start_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from factories import FakeBackend

    from llm_bench import cli

    monkeypatch.setattr(cli, "_backend", lambda url: FakeBackend())
    runner = CliRunner()
    run = [
        "run",
        "--model",
        "qwen",
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--no-telemetry",
        "--output-dir",
        str(tmp_path),
    ]
    export = ["export", "--input", str(tmp_path), "--output-dir", str(tmp_path / "exports")]
    assert runner.invoke(app, run).exit_code == 0
    first_ids = {e.experiment_id for e in load_experiments([tmp_path])}
    assert runner.invoke(app, export).exit_code == 0
    assert runner.invoke(app, ["clean", "--output-dir", str(tmp_path)]).exit_code == 0
    assert not list(tmp_path.glob("*.json"))
    assert runner.invoke(app, run).exit_code == 0
    assert runner.invoke(app, export).exit_code == 0
    snapshots = list((tmp_path / "exports").glob("*/manifest.json"))
    assert len(snapshots) == 1
    new_ids = set(json.loads(snapshots[0].read_text())["experiment_ids"])
    assert new_ids and not first_ids.intersection(new_ids)


def test_legacy_missing_timestamps_not_invented(tmp_path: Path) -> None:
    raw = json.loads(make_experiment().model_dump_json())
    raw.pop("timestamp_utc")
    raw["hardware"].pop("collected_at")
    raw.pop("experiment_id")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw))
    first = load_experiments([path])[0]
    second = load_experiments([path])[0]
    assert first == second
    assert first.timestamp_utc is None and first.hardware.collected_at is None
    export_snapshot([first], tmp_path / "snapshot")


def test_hardware_report_escapes_sensor_and_model_labels(tmp_path: Path) -> None:
    from llm_bench.report import generate_report

    experiment = fixture_experiment()
    experiment.model = "</script><script>alert(1)</script>"
    html = generate_report([experiment], tmp_path / "report.html").read_text()
    assert "</script><script>alert(1)" not in html
    assert "&lt;/script&gt;" in html
