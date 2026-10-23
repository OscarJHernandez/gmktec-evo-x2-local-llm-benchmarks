"""Explicit versioned Arrow schemas and immutable, self-contained export snapshots."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa
import pyarrow.parquet as pq

from llm_bench.models import Experiment, Sample
from llm_bench.observations import export_jsonl, export_observations
from llm_bench.report import generate_report

UTC = pa.timestamp("us", tz="UTC")
TEXT, INT, FLOAT, BOOL = pa.string(), pa.int64(), pa.float64(), pa.bool_()
CAP_FIELDS = [
    ("metric", TEXT),
    ("unit", TEXT),
    ("device", TEXT),
    ("source", TEXT),
    ("scope", TEXT),
    ("label", TEXT),
    ("kind", TEXT),
    ("available", BOOL),
    ("reason", TEXT),
    ("value_type", TEXT),
    ("counter_range", FLOAT),
]
CONFIG_FIELDS = [
    ("num_ctx", INT),
    ("num_predict", INT),
    ("temperature", FLOAT),
    ("seed", INT),
    ("think", BOOL),
    ("telemetry_enabled", BOOL),
    ("telemetry_interval_seconds", FLOAT),
    ("telemetry_host_role", TEXT),
]
SAMPLE_FIELDS = [
    ("timestamp_utc", UTC),
    ("started_at", UTC),
    ("ended_at", UTC),
    ("telemetry_duration_seconds", FLOAT),
    ("workload", TEXT),
    ("phase", TEXT),
    ("repetition", INT),
    ("requested_output_tokens", INT),
    ("prompt_tokens", INT),
    ("output_tokens", INT),
    ("generation_tokens_per_second", FLOAT),
    ("prompt_tokens_per_second", FLOAT),
    ("client_ttft_seconds", FLOAT),
    ("client_duration_seconds", FLOAT),
    ("server_load_seconds", FLOAT),
    ("server_total_seconds", FLOAT),
    ("prompt_eval_seconds", FLOAT),
    ("eval_seconds", FLOAT),
    ("prompt_cache_hit", BOOL),
    ("done_reason", TEXT),
    ("premature_completion", BOOL),
    ("client_metrics", BOOL),
    ("server_metrics", BOOL),
]
RESIDENCY = pa.struct(
    [
        ("name", TEXT),
        ("digest", TEXT),
        ("size", INT),
        ("size_vram", INT),
        ("context_length", INT),
        ("resident", BOOL),
    ]
)
HARDWARE = pa.struct(
    [
        ("collected_at", UTC),
        ("platform", TEXT),
        ("python_version", TEXT),
        ("cpu", TEXT),
        ("logical_cpu_count", INT),
        ("total_ram_bytes", INT),
        ("available_ram_bytes", INT),
        ("ram_percent", FLOAT),
        ("gpu", TEXT),
        ("gpu_memory_bytes", INT),
        ("gpu_memory_source", TEXT),
    ]
)
DEVICE = pa.struct([(key, TEXT) for key in ("device", "name", "vendor_id", "device_id", "driver")])
IDS = [("experiment_id", TEXT), ("invocation_id", TEXT)]
SCHEMAS = {
    "experiments": pa.schema(
        [
            ("experiment_id", TEXT),
            ("schema_version", TEXT),
            ("timestamp_utc", UTC),
            ("model", TEXT),
            ("model_digest", TEXT),
            ("quantization", TEXT),
            ("model_size_bytes", INT),
            ("parameter_size", TEXT),
            ("backend", TEXT),
            ("backend_version", TEXT),
            ("framework_version", TEXT),
            ("synthetic", BOOL),
            ("warmup_count", INT),
            ("repetition_count", INT),
            *CONFIG_FIELDS,
            ("hardware", HARDWARE),
            ("devices", pa.list_(DEVICE)),
            ("capabilities", pa.list_(pa.struct(CAP_FIELDS))),
            (
                "workloads",
                pa.list_(pa.struct([("name", TEXT), ("prompt", TEXT), ("output_limit", INT)])),
            ),
            ("warnings", pa.list_(TEXT)),
            ("settings_json", TEXT),
            (
                "sampling_stats",
                pa.struct(
                    [
                        ("collection_count", INT),
                        ("collection_seconds", FLOAT),
                        ("max_collection_seconds", FLOAT),
                        ("missed_deadlines", INT),
                        ("errors", pa.list_(TEXT)),
                    ]
                ),
            ),
        ]
    ),
    "inference_runs": pa.schema(
        [*IDS, *SAMPLE_FIELDS, ("residency_before", RESIDENCY), ("residency_after", RESIDENCY)]
    ),
    "telemetry": pa.schema(
        [
            *IDS,
            *CAP_FIELDS,
            ("sequence", INT),
            ("timestamp_utc", UTC),
            ("offset_seconds", FLOAT),
            ("value_int", INT),
            ("value_float", FLOAT),
            ("value_bool", BOOL),
        ]
    ),
    "telemetry_summary": pa.schema(
        [
            *IDS,
            *CAP_FIELDS,
            ("valid_count", INT),
            ("missing_count", INT),
            ("minimum", FLOAT),
            ("mean", FLOAT),
            ("median", FLOAT),
            ("maximum", FLOAT),
            ("standard_deviation", FLOAT),
            ("coverage_seconds", FLOAT),
            ("coverage_fraction", FLOAT),
            ("energy_joules", FLOAT),
            ("partial_energy_joules", FLOAT),
            ("joules_per_generated_token", FLOAT),
            ("energy_method", TEXT),
        ]
    ),
}
SCHEMAS = {
    name: schema.with_metadata({b"llm_bench_schema_version": b"1.2"})
    for name, schema in SCHEMAS.items()
}


def invocation_id(experiment: Experiment, sample: Sample, position: int) -> str:
    return (
        sample.invocation_id
        or uuid5(NAMESPACE_URL, f"{experiment.experiment_id}:sample:{position}").hex
    )


def unique_experiments(experiments: list[Experiment]) -> list[Experiment]:
    unique: dict[str, Experiment] = {}
    for experiment in experiments:
        prior = unique.get(experiment.experiment_id)
        if prior is not None and prior != experiment:
            raise ValueError(f"Conflicting experiment ID: {experiment.experiment_id}")
        unique[experiment.experiment_id] = experiment
    return list(unique.values())


def analytical_rows(experiments: list[Experiment]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SCHEMAS}
    seen_runs: set[str] = set()
    for experiment in unique_experiments(experiments):
        data = experiment.model_dump()
        data.update({key: experiment.settings.get(key) for key, _ in CONFIG_FIELDS})
        data["workloads"] = experiment.settings.get("workloads", [])
        data["settings_json"] = json.dumps(experiment.settings, allow_nan=False, sort_keys=True)
        rows["experiments"].append(data)
        run_ids = set()
        for position, sample in enumerate(experiment.samples):
            run_id = invocation_id(experiment, sample, position)
            if run_id in seen_runs:
                raise ValueError(f"Duplicate invocation ID: {run_id}")
            seen_runs.add(run_id)
            run_ids.add(run_id)
            rows["inference_runs"].append(
                {
                    **sample.model_dump(),
                    "experiment_id": experiment.experiment_id,
                    "invocation_id": run_id,
                }
            )
        for name, observations in (
            ("telemetry", experiment.telemetry),
            ("telemetry_summary", experiment.telemetry_summaries),
        ):
            keys: set[tuple[Any, ...]] = set()
            for item in observations:
                if item.invocation_id not in run_ids:
                    raise ValueError(f"Orphan telemetry invocation: {item.invocation_id}")
                key = (
                    item.invocation_id,
                    item.metric,
                    item.device,
                    item.source,
                    getattr(item, "sequence", None),
                )
                if key in keys:
                    raise ValueError(f"Duplicate {name} observation: {key}")
                keys.add(key)
                rows[name].append({**item.model_dump(), "experiment_id": experiment.experiment_id})
    return rows


def export_snapshot(experiments: list[Experiment], output: Path) -> Path:
    experiments = unique_experiments(experiments)
    if not experiments:
        raise ValueError("No experiments to export")
    rows = analytical_rows(experiments)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation; never overwrite an existing snapshot.
    staging = Path(tempfile.mkdtemp(prefix=".benchmark-export-", dir=output.parent))
    try:
        for name, schema in SCHEMAS.items():
            table = pa.Table.from_pylist(rows[name], schema=schema)
            pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
        export_observations(experiments, staging / "observations.csv")
        export_jsonl(experiments, staging / "observations.jsonl")
        generate_report(experiments, staging / "report.html")
        manifest = {
            "format": "llm-bench-export",
            "schema_version": "1.2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_ids": [e.experiment_id for e in experiments],
            "row_counts": {name: len(values) for name, values in rows.items()},
            "files": sorted(p.name for p in staging.iterdir()),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        output.rmdir()
        raise
    return output
