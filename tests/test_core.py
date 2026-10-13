from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_bench.backends.ollama import GenerateResult, OllamaBackend, OllamaError
from llm_bench.models import BenchmarkConfig, Experiment, HardwareInfo, Sample, summarize_samples
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
    assert _duration_ns("bad") is None


def test_aggregation_preserves_runs_and_handles_single_sample() -> None:
    samples = [
        Sample(workload="short", phase="benchmark", repetition=index, requested_output_tokens=10,
               prompt_tokens=5, output_tokens=10, generation_tokens_per_second=value,
               prompt_tokens_per_second=2, client_ttft_seconds=1, client_duration_seconds=2)
        for index, value in enumerate([3.0, 5.0], 1)
    ]
    summary = summarize_samples(samples)["generation_tokens_per_second"]
    assert summary.count == 2
    assert summary.mean == 4
    assert summary.median == 4
    assert summary.standard_deviation == pytest.approx(2**0.5)
    assert summary.samples == [3.0, 5.0]


def test_ollama_stream_parse_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def __iter__(self):
            return iter([
                b'{"response":"","done":false}\n',
                b'{"response":"hello","done":false}\n',
                b'{"response":" world","done":true,"eval_count":2}\n',
            ])

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


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def health(self) -> str:
        return "test"

    def find_model(self, name: str) -> dict[str, Any]:
        return {"name": name, "digest": "sha256:fake", "details": {"quantization_level": "Q4_K_M"}}

    def generate(
        self, model: str, prompt: str, *, options: dict[str, Any], think: bool,
        keep_alive: str | int | None = None,
    ) -> GenerateResult:
        self.calls += 1
        return GenerateResult(
            {
                "prompt_eval_count": 10,
                "eval_count": 0 if self.calls == 1 else 16,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 2_000_000_000,
                "load_duration": 50_000_000,
                "total_duration": 3_000_000_000,
                "prompt_cache_hit": False,
                "done": True,
            },
            2.2,
            0.4,
        )


def test_runner_records_server_and_client_measurements_and_zero_tokens() -> None:
    backend = FakeBackend()
    result = run_benchmark(BenchmarkConfig(model="qwen", repeat=1, warmup=0), backend=backend)  # type: ignore[arg-type]
    assert len(result.samples) == 3
    assert result.samples[0].output_tokens == 0
    assert result.samples[0].generation_tokens_per_second is None
    assert result.samples[1].generation_tokens_per_second == 8
    assert result.samples[1].prompt_tokens_per_second == 20
    assert result.samples[1].client_ttft_seconds == 0.4
    assert result.model_digest == "sha256:fake"


def make_experiment() -> Experiment:
    sample = Sample(
        workload="short-input", phase="benchmark", repetition=1, requested_output_tokens=12,
        prompt_tokens=10, output_tokens=12, generation_tokens_per_second=6.0,
        prompt_tokens_per_second=5.0, client_ttft_seconds=0.2, client_duration_seconds=2.0,
    )
    return Experiment(
        framework_version="0.1.0", model="sample-model", model_digest="synthetic",
        hardware=HardwareInfo(platform="synthetic", python_version="3", cpu="synthetic"),
        settings={}, warmup_count=0, repetition_count=1, samples=[sample],
        summaries=summarize_samples([sample]), synthetic=True,
    )


def test_result_roundtrip_csv_and_offline_report(tmp_path: Path) -> None:
    experiment = make_experiment()
    json_path = save_experiment(experiment, tmp_path / "sample.json")
    loaded = load_experiments([json_path])
    assert loaded[0].samples[0].output_tokens == 12
    csv_path = export_csv(loaded, tmp_path / "samples.csv")
    assert "generation_tokens_per_second" in csv_path.read_text()
    report_path = generate_report(loaded, tmp_path / "report.html")
    html = report_path.read_text()
    assert "SYNTHETIC EXAMPLE DATA" in html
    assert "Plotly.newPlot" in html


def test_model_names_and_malformed_api_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaBackend("http://localhost:11434")
    monkeypatch.setattr(backend, "_request", lambda path: {"models": "bad"})
    with pytest.raises(OllamaError, match="Malformed"):
        backend.models()
