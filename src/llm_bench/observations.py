"""Append-only, analysis-ready observation exports. No statistical assumptions imposed."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm_bench.models import Experiment

METRICS = (
    "prompt_tokens",
    "output_tokens",
    "requested_output_tokens",
    "generation_tokens_per_second",
    "prompt_tokens_per_second",
    "client_ttft_seconds",
    "client_duration_seconds",
    "server_load_seconds",
    "server_total_seconds",
    "prompt_eval_seconds",
    "eval_seconds",
)


def observation_rows(experiments: list[Experiment]) -> list[dict[str, Any]]:
    """Return one tidy row per measured metric per repetition."""
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        config = experiment.settings
        for sample in experiment.samples:
            for metric in METRICS:
                value = getattr(sample, metric)
                rows.append(
                    {
                        "experiment_id": experiment.experiment_id,
                        "timestamp_utc": experiment.timestamp_utc.isoformat(),
                        "model": experiment.model,
                        "model_digest": experiment.model_digest,
                        "quantization": experiment.quantization,
                        "backend": experiment.backend,
                        "backend_version": experiment.backend_version,
                        "framework_version": experiment.framework_version,
                        "workload": sample.workload,
                        "phase": sample.phase,
                        "repetition": sample.repetition,
                        "metric": metric,
                        "value": value,
                        "sample_timestamp_utc": sample.timestamp_utc.isoformat()
                        if sample.timestamp_utc
                        else None,
                        "prompt_cache_hit": sample.prompt_cache_hit,
                        "premature_completion": sample.premature_completion,
                        "done_reason": sample.done_reason,
                        "num_ctx": config.get("num_ctx"),
                        "num_predict": config.get("num_predict"),
                        "temperature": config.get("temperature"),
                        "think": config.get("think"),
                        "seed": config.get("seed"),
                        "gpu": experiment.hardware.gpu,
                            "total_ram_bytes": experiment.hardware.total_ram_bytes,
                    }
                )
    return rows


def export_observations(experiments: list[Experiment], output: Path) -> Path:
    """Create a tidy CSV for pandas/PyMC. Never silently append duplicate rows."""
    rows = observation_rows(experiments)
    if not rows:
        raise ValueError("No observations to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing observations: {output}")
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def export_jsonl(experiments: list[Experiment], output: Path) -> Path:
    """Create durable JSONL with native null values and one record per metric."""
    rows = observation_rows(experiments)
    if not rows:
        raise ValueError("No observations to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return output
