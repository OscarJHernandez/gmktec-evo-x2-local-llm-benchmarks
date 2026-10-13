"""Command-line interface."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_bench.backends.base import InferenceBackend
from llm_bench.backends.ollama import OllamaBackend, OllamaError
from llm_bench.models import BenchmarkConfig
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
) -> None:
    """Run default short-input, long-input, and long-generation workloads."""
    _run(model, repeat, warmup, api_url, output_dir, csv)


@app.command()
def compare(
    models: str = typer.Option(..., help="Comma-separated installed Ollama model names."),
    repeat: int = typer.Option(5, min=1, max=100, help="Measured repetitions per workload."),
    warmup: int = typer.Option(1, min=0, max=20, help="Warm-up repetitions per workload."),
    api_url: str = typer.Option(DEFAULT_API, "--api-url", help="Ollama API URL."),
    output_dir: Path = typer.Option(
        DEFAULT_RESULTS, help="Directory for machine-readable results."
    ),
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
            _run(name, repeat, warmup, api_url, output_dir, False, backend=backend)
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    output: Path = typer.Option(Path("report.html"), "--output", help="Output HTML report path."),
    input_dir: Path = typer.Option(
        DEFAULT_RESULTS, "--input", help="JSON file or results directory."
    ),
    csv_output: Path | None = typer.Option(None, "--csv", help="Optional CSV output path."),
) -> None:
    """Create an offline HTML report from saved experiment JSON files."""
    try:
        experiments = load_experiments([input_dir])
        if not experiments:
            raise typer.BadParameter(f"No result JSON files found in {input_dir}.")
        generate_report(experiments, output)
        if csv_output:
            export_csv(experiments, csv_output)
    except (OSError, ValueError) as exc:
        console.print(f"[red]Could not generate report: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Report saved:[/green] {output}")
    if csv_output:
        console.print(f"[green]CSV saved:[/green] {csv_output}")


def _run(
    model: str,
    repeat: int,
    warmup: int,
    api_url: str,
    output_dir: Path,
    csv: bool,
    *,
    backend: InferenceBackend | None = None,
) -> None:
    try:
        config = BenchmarkConfig(model=model, repeat=repeat, warmup=warmup, api_url=api_url)
        # Check model availability before starting a potentially lengthy experiment.
        client = backend or _backend(api_url)
        client.find_model(model)
        experiment = run_benchmark(config, client)
    except (ValueError, OllamaError) as exc:
        console.print(f"[red]Benchmark could not start: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    safe_model = "".join(char if char.isalnum() or char in "-_." else "_" for char in model)
    output = output_dir / f"{safe_model}-{experiment.timestamp_utc:%Y%m%dT%H%M%SZ}.json"
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


if __name__ == "__main__":
    app()
