"""JSON and CSV result persistence."""

from __future__ import annotations

import csv
from pathlib import Path

from llm_bench.models import Experiment


def save_experiment(experiment: Experiment, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(experiment.model_dump_json(indent=2) + "\n", encoding="utf-8")
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
            experiments.append(Experiment.model_validate_json(file.read_text(encoding="utf-8")))
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
