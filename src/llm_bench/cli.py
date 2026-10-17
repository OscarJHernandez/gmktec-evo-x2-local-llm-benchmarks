"""Command-line interface."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from llm_bench.backends.base import InferenceBackend
from llm_bench.backends.ollama import OllamaBackend, OllamaError
from llm_bench.model_catalog import (
    catalog_candidates,
    fit_reason,
    free_storage_bytes,
    model_storage_dir,
    select_installed,
    system_ram_bytes,
)
from llm_bench.models import BenchmarkConfig
from llm_bench.observations import export_jsonl, export_observations
from llm_bench.physics import DATASETS, prepare as prepare_physics
from llm_bench.physics import run as run_physics
from llm_bench.physics import summary as physics_summary
from llm_bench.persistence import export_csv, load_experiments, save_experiment
from llm_bench.report import generate_report
from llm_bench.runner import run_benchmark

app = typer.Typer(no_args_is_help=True, help="Reproducible benchmarks for local language models.")
console = Console()
DEFAULT_API = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_RESULTS = Path(os.getenv("LLM_BENCH_RESULTS_DIR", "results"))


def _backend(api_url: str) -> OllamaBackend:
    return OllamaBackend(api_url)


@app.command()
def doctor(api_url: str = typer.Option(DEFAULT_API, "--api-url", help="Ollama API URL.")) -> None:
    """Check Python, system resources, and Ollama connectivity."""
    import platform

    from llm_bench.hardware import collect_hardware

    console.print(f"[bold]Python[/bold] {platform.python_version()}")
    info = collect_hardware()
    console.print(f"[bold]CPU[/bold] {info.cpu}; logical cores: {info.logical_cpu_count}")
    console.print(
        f"[bold]RAM[/bold] {info.total_ram_bytes or 0:,} bytes; "
        f"{info.ram_percent if info.ram_percent is not None else 'unknown'}% in use"
    )
    console.print(f"[bold]GPU[/bold] {info.gpu or 'not detected (optional)'}")
    try:
        console.print(f"[green]Ollama[/green] {_backend(api_url).health()} at configured endpoint")
    except OllamaError as exc:
        console.print(f"[red]Ollama unavailable[/red]: {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def models(api_url: str = typer.Option(DEFAULT_API, "--api-url", help="Ollama API URL.")) -> None:
    """List models already installed in Ollama (does not download models)."""
    try:
        found = _backend(api_url).models()
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table("Model", "Digest", "Quantization", "Size")
    for item in found:
        details = item.get("details") or {}
        table.add_row(
            str(item.get("name", item.get("model", "unknown"))),
            str(item.get("digest", "unknown")),
            str(details.get("quantization_level", "unknown")),
            str(item.get("size", "unknown")),
        )
    console.print(table)


@app.command()
def run(
    model: str = typer.Option(..., help="Installed Ollama model name."),
    repeat: int = typer.Option(5, min=1, max=100, help="Measured repetitions per workload."),
    warmup: int = typer.Option(1, min=0, max=20, help="Warm-up repetitions per workload."),
    api_url: str = typer.Option(DEFAULT_API, "--api-url", help="Ollama API URL."),
    output_dir: Path = typer.Option(
        DEFAULT_RESULTS, help="Directory for machine-readable results."
    ),
    csv: bool = typer.Option(False, "--csv", help="Also export samples to CSV."),
    num_ctx: int = typer.Option(4096, min=1),
    num_predict: int = typer.Option(128, min=1),
    think: bool = typer.Option(False, "--think/--no-think"),
) -> None:
    """Run default short-input, long-input, and long-generation workloads."""
    _run(
        model,
        repeat,
        warmup,
        api_url,
        output_dir,
        csv,
        num_ctx=num_ctx,
        num_predict=num_predict,
        think=think,
    )


@app.command()
def compare(
    models: str = typer.Option(..., help="Comma-separated installed Ollama model names."),
    repeat: int = typer.Option(5, min=1, max=100, help="Measured repetitions per workload."),
    warmup: int = typer.Option(1, min=0, max=20, help="Warm-up repetitions per workload."),
    api_url: str = typer.Option(DEFAULT_API, "--api-url", help="Ollama API URL."),
    output_dir: Path = typer.Option(
        DEFAULT_RESULTS, help="Directory for machine-readable results."
    ),
    num_ctx: int = typer.Option(4096, min=1),
    num_predict: int = typer.Option(128, min=1),
    think: bool = typer.Option(False, "--think/--no-think"),
) -> None:
    """Benchmark multiple installed models sequentially and compare results."""
    names = list(dict.fromkeys(name.strip() for name in models.split(",") if name.strip()))
    if not names:
        raise typer.BadParameter("Provide at least one comma-separated model name.")
    backend = _backend(api_url)
    try:
        installed = backend.models()
        available = {str(item.get("name") or item.get("model")) for item in installed}
        missing = [
            name for name in names if name not in available and f"{name}:latest" not in available
        ]
        if missing:
            raise OllamaError(
                f"Models not installed: {', '.join(missing)}. No models were downloaded."
            )
        for name in names:
            _run(
                name,
                repeat,
                warmup,
                api_url,
                output_dir,
                False,
                backend=backend,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=think,
            )
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("catalog-run")
def catalog_run(
    repeat: int = typer.Option(50, min=1, max=100),
    warmup: int = typer.Option(3, min=0, max=20),
    api_url: str = typer.Option(DEFAULT_API, "--api-url"),
    output_dir: Path = typer.Option(DEFAULT_RESULTS, "--output-dir"),
    num_ctx: int = typer.Option(4096, min=1),
    num_predict: int = typer.Option(128, min=1),
    think: bool = typer.Option(False, "--think/--no-think"),
    auto_pull: bool = typer.Option(False, "--auto-pull/--no-auto-pull"),
    installed_only: bool = typer.Option(False, "--installed-only"),
    catalog_file: Path | None = typer.Option(None, "--catalog-file"),
    model_dir: Path | None = typer.Option(None, "--model-dir"),
) -> None:
    """Benchmark the curated fit-checked Ollama catalog plus installed models.

    Use --auto-pull to download missing models explicitly. Results and progress
    are stored locally and each failed model does not block the next.
    """
    if auto_pull and installed_only:
        raise typer.BadParameter("Do not combine --auto-pull with --installed-only")
    if urlparse(api_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise typer.BadParameter(
            "Catalog fit checks require a local Ollama server. Run on the Ollama host."
        )

    try:
        client = _backend(api_url)
        installed = client.models()
        candidates = (
            select_installed(installed) if installed_only else catalog_candidates(catalog_file)
        )
        if not installed_only:
            known_names = {candidate.name for candidate in candidates}
            candidates.extend(
                item for item in select_installed(installed) if item.name not in known_names
            )
        total_ram = system_ram_bytes()
        store = model_storage_dir(model_dir)
        free_storage_bytes(store)
    except (OllamaError, OSError, ValueError) as exc:
        console.print(f"[red]Catalog preflight failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    installed_by_name = {str(item.get("name") or item.get("model")): item for item in installed}
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = output_dir / (
        f"catalog-progress-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex}.jsonl"
    )
    successful = 0
    skipped = 0
    failed = 0
    with progress.open("x", encoding="utf-8") as stream:
        for index, candidate in enumerate(candidates, 1):
            name = candidate.name
            metadata = installed_by_name.get(name)
            actual_size = metadata.get("size") if metadata else None
            size_gb = (
                actual_size / 1_000_000_000
                if isinstance(actual_size, int) and actual_size > 0
                else candidate.size_gb
            )
            from llm_bench.model_catalog import Candidate

            sized = Candidate(name, size_gb)
            try:
                free_disk = free_storage_bytes(store)
                reason = fit_reason(
                    sized,
                    total_ram_bytes=total_ram,
                    free_disk_bytes=free_disk,
                    already_installed=metadata is not None,
                )
            except OSError as exc:
                reason = f"Cannot verify Ollama model storage: {exc}"
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": name,
                "estimated_size_gb": size_gb,
                "candidate_index": index,
                "candidate_count": len(candidates),
                "installed_before": metadata is not None,
                "status": "pending",
                "reason": None,
            }
            if metadata is None and not auto_pull:
                record["status"], record["reason"] = (
                    "skipped_not_installed",
                    "Enable --auto-pull to download candidate",
                )
            elif reason is not None:
                record["status"], record["reason"] = "skipped_fit", reason
            else:
                try:
                    if metadata is None:
                        console.print(f"[cyan]Pulling[/cyan] {name} ({index}/{len(candidates)})")
                        client.pull(name)
                        metadata = client.find_model(name)
                    console.print(f"[cyan]Benchmarking[/cyan] {name} ({index}/{len(candidates)})")
                    _run(
                        name,
                        repeat,
                        warmup,
                        api_url,
                        output_dir,
                        False,
                        backend=client,
                        num_ctx=num_ctx,
                        num_predict=num_predict,
                        think=think,
                    )
                    record["status"] = "completed"
                    successful += 1
                except (OllamaError, typer.Exit, OSError, ValueError) as exc:
                    record["status"], record["reason"] = (
                        "failed",
                        str(exc) if not isinstance(exc, typer.Exit) else "Benchmark failed",
                    )
                    failed += 1
            if str(record["status"]).startswith("skipped"):
                skipped += 1
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            console.print(
                f"{name}: {record['status']}"
                + (f" ({record['reason']})" if record["reason"] else "")
            )
    console.print(
        f"[bold]Catalog complete[/bold]: {successful} passed, "
        f"{skipped} skipped, {failed} failed. Status records: {progress}"
    )
    if successful == 0:
        raise typer.Exit(code=1)


@app.command()
def report(
    output: Path = typer.Option(Path("report.html"), "--output", help="Output HTML report path."),
    input_dir: Path = typer.Option(
        DEFAULT_RESULTS, "--input", help="JSON file or results directory."
    ),
    csv_output: Path | None = typer.Option(None, "--csv", help="Optional CSV output path."),
    observations: Path | None = typer.Option(
        None, "--observations", help="Tidy metric-per-row CSV for Bayesian analysis."
    ),
    jsonl: Path | None = typer.Option(
        None, "--jsonl", help="Tidy metric-per-row JSONL for Bayesian analysis."
    ),
) -> None:
    """Create an offline HTML report from saved experiment JSON files."""
    try:
        experiments = load_experiments([input_dir])
        if not experiments:
            raise typer.BadParameter(f"No result JSON files found in {input_dir}.")
        generate_report(experiments, output)
        if csv_output:
            export_csv(experiments, csv_output)
        if observations:
            export_observations(experiments, observations)
        if jsonl:
            export_jsonl(experiments, jsonl)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not generate report: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Report saved:[/green] {output}")
    if csv_output:
        console.print(f"[green]CSV saved:[/green] {csv_output}")
    if observations:
        console.print(f"[green]Observations saved:[/green] {observations}")
    if jsonl:
        console.print(f"[green]JSONL saved:[/green] {jsonl}")


def _run(
    model: str,
    repeat: int,
    warmup: int,
    api_url: str,
    output_dir: Path,
    csv: bool,
    *,
    backend: InferenceBackend | None = None,
    num_ctx: int = 4096,
    num_predict: int = 128,
    think: bool = False,
) -> None:
    try:
        config = BenchmarkConfig(
            model=model,
            repeat=repeat,
            warmup=warmup,
            api_url=api_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
            think=think,
        )
        # Check model availability before starting a potentially lengthy experiment.
        client = backend or _backend(api_url)
        client.find_model(model)
        experiment = run_benchmark(config, client)
    except (ValueError, OllamaError) as exc:
        console.print(f"[red]Benchmark could not start: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    safe_model = "".join(char if char.isalnum() or char in "-_." else "_" for char in model)
    output = (
        output_dir
        / f"{safe_model}-{experiment.timestamp_utc:%Y%m%dT%H%M%SZ}-{experiment.experiment_id}.json"
    )
    save_experiment(experiment, output)
    table = Table("Workload", "Median output tok/s", "Median client TTFT", "Runs")
    for workload in sorted({item.workload for item in experiment.samples}):
        values = [
            item.generation_tokens_per_second
            for item in experiment.samples
            if item.workload == workload
            and item.phase in {"benchmark", "warm_start"}
            and item.generation_tokens_per_second is not None
        ]
        ttft = [
            item.client_ttft_seconds
            for item in experiment.samples
            if item.workload == workload
            and item.phase in {"benchmark", "warm_start"}
            and item.client_ttft_seconds is not None
        ]
        from statistics import median

        table.add_row(
            workload,
            f"{median(values):.2f}" if values else "n/a",
            f"{median(ttft):.3f}s" if ttft else "n/a",
            str(len(values)),
        )
    console.print(table)
    console.print(f"Saved raw measurements: {output}")
    if experiment.warnings:
        console.print(f"[yellow]{len(experiment.warnings)} warning(s); see result JSON.[/yellow]")
    if csv:
        csv_path = output.with_suffix(".csv")
        export_csv([experiment], csv_path)
        console.print(f"Saved CSV: {csv_path}")


@app.command("physics-prepare")
def physics_prepare(
    dataset: str = typer.Option("scibench", "--dataset", help="scibench, ugphysics, phybench"),
    data_dir: Path = typer.Option(DEFAULT_RESULTS / "datasets", "--data-dir"),
) -> None:
    """Download a selected physics dataset once from its official public repository."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASETS)}")
    try:
        paths = prepare_physics(dataset, data_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]Physics dataset download failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Physics dataset {dataset}: {len(paths)} cached source file(s) in {data_dir}")


@app.command("physics-run")
def physics_benchmark(
    dataset: str = typer.Option("scibench", "--dataset"),
    models: str = typer.Option("installed", "--models", help="installed or comma-separated tags"),
    repeat: int = typer.Option(3, "--repeat", min=1, max=100),
    max_questions: int = typer.Option(50, "--max-questions", min=1),
    seed: int = typer.Option(42, "--seed"),
    temperature: float = typer.Option(0.3, "--temperature", min=0.0, max=2.0),
    num_ctx: int = typer.Option(8192, "--num-ctx", min=1),
    num_predict: int = typer.Option(1024, "--num-predict", min=1),
    think: bool = typer.Option(False, "--think/--no-think"),
    fetch: bool = typer.Option(False, "--fetch/--no-fetch", help="Explicitly fetch dataset"),
    data_dir: Path = typer.Option(DEFAULT_RESULTS / "datasets", "--data-dir"),
    output_dir: Path = typer.Option(DEFAULT_RESULTS / "physics", "--output-dir"),
    api_url: str = typer.Option(DEFAULT_API, "--api-url"),
) -> None:
    """Run optional physics reasoning evaluation; append resumable raw JSONL."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASETS)}")
    names = [name.strip() for name in models.split(",") if name.strip()]
    if not names or ("installed" in names and len(names) != 1):
        raise typer.BadParameter("Use 'installed' or comma-separated model tags")
    output = output_dir / f"{dataset}-attempts.jsonl"
    try:
        if fetch:
            prepare_physics(dataset, data_dir)
        counts = run_physics(
            dataset=dataset,
            models=names,
            root=data_dir,
            output=output,
            repeat=repeat,
            max_questions=max_questions,
            seed=seed,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            think=think,
            api_url=api_url,
        )
    except (OSError, ValueError, OllamaError, RuntimeError) as exc:
        console.print(f"[red]Physics benchmark failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Physics attempts: {counts}. Raw data: {output}")
    console.print("Scores are a local numeric proxy, not official benchmark scores.")


@app.command("physics-results")
def physics_results(
    dataset: str = typer.Option("scibench", "--dataset"),
    output_dir: Path = typer.Option(DEFAULT_RESULTS / "physics", "--output-dir"),
) -> None:
    """Summarize local numerical correctness; retain all raw scored/unscored attempts."""
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASETS)}")
    path = output_dir / f"{dataset}-attempts.jsonl"
    try:
        results = physics_summary(path)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not read physics results: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(
        "Model",
        "Attempts",
        "Score eligible",
        "Correct",
        "Unscored",
        "Inference errors",
        "Numeric proxy accuracy",
    )
    for model, row in sorted(results.items()):
        accuracy = row["accuracy_numeric_proxy"]
        table.add_row(
            model,
            str(row["attempts"]),
            str(row["scored"]),
            str(row["correct"]),
            str(row["unscored"]),
            str(row["inference_errors"]),
            f"{accuracy:.2%}" if isinstance(accuracy, float) else "n/a",
        )
    console.print(table)
    console.print(f"All attempts (including complete responses): {path}")


if __name__ == "__main__":
    app()
