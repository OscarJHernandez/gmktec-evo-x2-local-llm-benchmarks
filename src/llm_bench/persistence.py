"""JSON and CSV result persistence."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from llm_bench.models import Experiment


def save_experiment(experiment: Experiment, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".experiment-", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(experiment.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic publication, exclusive destination.
    finally:
        os.unlink(temporary)
    return output


def load_experiments(paths: list[Path]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for path in paths:
        if path.is_dir():
            files = sorted(path.glob("*.json"))
        else:
            files = [path]
        for file in files:
            if file.suffix != ".json":
                continue
            raw = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Expected experiment object: {file}")
            raw.setdefault(
                "experiment_id",
                uuid5(NAMESPACE_URL, json.dumps(raw, sort_keys=True, separators=(",", ":"))).hex,
            )
            raw.setdefault("schema_version", "1.0")
            raw.setdefault("timestamp_utc", None)
            if isinstance(raw.get("hardware"), dict):
                raw["hardware"].setdefault("collected_at", None)
            for position, sample in enumerate(raw.get("samples", [])):
                if not sample.get("invocation_id"):
                    sample["invocation_id"] = uuid5(
                        NAMESPACE_URL, f"{raw['experiment_id']}:sample:{position}"
                    ).hex
            experiments.append(Experiment.model_validate(raw))
    return experiments


def export_csv(experiments: list[Experiment], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "model_digest",
        "workload",
        "phase",
        "repetition",
        "prompt_tokens",
        "output_tokens",
        "generation_tokens_per_second",
        "prompt_tokens_per_second",
        "client_ttft_seconds",
        "client_duration_seconds",
        "server_load_seconds",
        "prompt_cache_hit",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for experiment in experiments:
            for sample in experiment.samples:
                writer.writerow(
                    {
                        "model": experiment.model,
                        "model_digest": experiment.model_digest,
                        **sample.model_dump(include=set(columns) - {"model", "model_digest"}),
                    }
                )
    return output
