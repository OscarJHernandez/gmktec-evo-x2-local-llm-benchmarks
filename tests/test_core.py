from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from factories import FakeBackend, make_experiment
from typer.testing import CliRunner

from llm_bench import cli
from llm_bench.backends.ollama import OllamaBackend, OllamaError
from llm_bench.models import BenchmarkConfig, Experiment, Sample, summarize_samples
from llm_bench.persistence import export_csv, load_experiments, save_experiment
from llm_bench.report import generate_report
from llm_bench.runner import run_benchmark


def test_config_rejects_path_and_invalid_repetition() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(model="../private")
    with pytest.raises(ValueError):
        BenchmarkConfig(model="qwen", repeat=0)


def test_duration_zero_and_bad_values() -> None:
    from llm_bench.runner import _duration_ns

    assert _duration_ns(0) == 0
    assert _duration_ns(2_000_000_000) == 2
    assert _duration_ns(-1) is None
    assert _duration_ns(True) is None
    assert _duration_ns(float("inf")) is None
    assert _duration_ns("bad") is None


def test_aggregation_preserves_runs_and_handles_single_sample() -> None:
    samples = [
        Sample(
            workload="short",
            phase="warm_start",
            repetition=index,
            requested_output_tokens=10,
            prompt_tokens=5,
            output_tokens=10,
            generation_tokens_per_second=value,
            prompt_tokens_per_second=2,
            client_ttft_seconds=1,
            client_duration_seconds=2,
        )
        for index, value in enumerate([3.0, 5.0], 1)
    ]
    summary = summarize_samples(samples)["generation_tokens_per_second"]
    assert summary.count == 2
    assert summary.mean == 4
    assert summary.median == 4
    assert summary.standard_deviation == pytest.approx(2**0.5)
    assert summary.samples == [3.0, 5.0]


def test_prompt_cache_aggregates_are_not_mixed() -> None:
    samples = [
        Sample(
            workload="long-input",
            phase="warm_start",
            repetition=index,
            requested_output_tokens=10,
            prompt_tokens=100,
            output_tokens=10,
            generation_tokens_per_second=5,
            prompt_tokens_per_second=20,
            client_ttft_seconds=1,
            client_duration_seconds=2,
            prompt_cache_hit=cache_hit,
        )
        for index, cache_hit in enumerate((False, True, None), 1)
    ]
    summaries = summarize_samples(samples)
    assert summaries["prompt_tokens_per_second_uncached"].samples == [20.0]
    assert summaries["prompt_tokens_per_second_cached"].samples == [20.0]
    assert summaries["prompt_tokens_per_second_cache_unknown"].samples == [20.0]
    assert summaries["workload:long-input:prompt_tokens_per_second_uncached"].count == 1


def test_ollama_stream_parse_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def __iter__(self):
            return iter(
                [
                    b'{"response":"","done":false}\n',
                    b'{"response":"hello","done":false}\n',
                    b'{"response":" world","done":true,"eval_count":2}\n',
                ]
            )

    monkeypatch.setattr("llm_bench.backends.ollama.urlopen", lambda *a, **k: Response())
    result = OllamaBackend("http://localhost:11434").generate("qwen", "hi", options={})
    assert result.payload["response"] == "hello world"
    assert result.payload["eval_count"] == 2
    assert result.client_ttft_seconds is not None

    class InvalidResponse(Response):
        def __iter__(self):
            return iter([b'{"response":"not done","done":false}\n'])

    monkeypatch.setattr("llm_bench.backends.ollama.urlopen", lambda *a, **k: InvalidResponse())
    with pytest.raises(OllamaError, match="without a final"):
        OllamaBackend("http://localhost:11434").generate("qwen", "hi", options={})


def test_runner_records_server_and_client_measurements_and_zero_tokens() -> None:
    backend = FakeBackend()
    result = run_benchmark(
        BenchmarkConfig(model="qwen", repeat=1, warmup=0),
        backend=backend,
        gpu_collector=lambda: ("Synthetic GPU", 123, "test collector"),
    )
    assert len(result.samples) == 4
    assert result.samples[0].output_tokens == 0
    assert result.samples[0].generation_tokens_per_second is None
    assert result.samples[0].phase == "cold_start"
    assert result.samples[1].phase == "warm_start"
    assert result.samples[1].generation_tokens_per_second == 8
    assert result.samples[1].prompt_tokens_per_second == 20
    assert result.samples[1].client_ttft_seconds == 0.4
    assert result.model_digest == "sha256:fake"
    assert result.summaries["cold_start_load_seconds"].samples == [0.05]
    assert result.hardware.gpu == "Synthetic GPU"
    assert result.hardware.gpu_memory_bytes == 123


def test_cli_run_persists_json_and_compare_validates_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(cli, "_backend", lambda api_url: backend)
    cli_runner = CliRunner()
    run_result = cli_runner.invoke(
        cli.app,
        [
            "run",
            "--model",
            "qwen",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    results = list(tmp_path.glob("*.json"))
    assert len(results) == 1
    assert Experiment.model_validate_json(results[0].read_text()).model == "qwen"

    compare_result = cli_runner.invoke(
        cli.app,
        [
            "compare",
            "--models",
            "qwen,missing",
            "--repeat",
            "1",
            "--output-dir",
            str(tmp_path / "compare"),
        ],
    )
    assert compare_result.exit_code == 1
    assert not (tmp_path / "compare").exists()

    monkeypatch.setattr(
        backend,
        "models",
        lambda: [{"name": "qwen:latest"}, {"name": "mistral:latest"}],
    )
    compare_result = cli_runner.invoke(
        cli.app,
        [
            "compare",
            "--models",
            "qwen,mistral",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--output-dir",
            str(tmp_path / "compare"),
        ],
    )
    assert compare_result.exit_code == 0, compare_result.output
    assert len(list((tmp_path / "compare").glob("*.json"))) == 2


def test_result_roundtrip_csv_and_offline_report(tmp_path: Path) -> None:
    experiment = make_experiment()
    experiment.model = "</script><script>alert(1)</script>"
    json_path = save_experiment(experiment, tmp_path / "sample.json")
    loaded = load_experiments([json_path])
    assert loaded[0].samples[0].output_tokens == 12
    csv_path = export_csv(loaded, tmp_path / "samples.csv")
    assert "generation_tokens_per_second" in csv_path.read_text()
    report_path = generate_report(loaded, tmp_path / "report.html")
    html = report_path.read_text()
    assert "SYNTHETIC EXAMPLE DATA" in html
    assert "known uncached only" in html
    assert "Plotly.newPlot" in html
    assert "</script><script>alert(1)" not in html
    assert "&lt;/script&gt;" in html


def test_model_names_and_malformed_api_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaBackend("http://localhost:11434")
    monkeypatch.setattr(backend, "_request", lambda path: {"models": "bad"})
    with pytest.raises(OllamaError, match="Malformed"):
        backend.models()
