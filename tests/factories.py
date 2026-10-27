"""Shared builders for tests: a fake backend and synthetic experiments/observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_bench.backends.base import GenerateResult
from llm_bench.models import Experiment, HardwareInfo, Sample, summarize_samples
from llm_bench.telemetry.aggregation import summarize_telemetry
from llm_bench.telemetry.collectors import capability
from llm_bench.telemetry.schemas import Observation


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def health(self) -> str:
        return "test"

    def find_model(self, name: str) -> dict[str, Any]:
        return {"name": name, "digest": "sha256:fake", "details": {"quantization_level": "Q4_K_M"}}

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any],
        think: bool,
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

    def models(self) -> list[dict[str, Any]]:
        return [{"name": "qwen:latest"}]


def make_experiment() -> Experiment:
    sample = Sample(
        workload="short-input",
        phase="benchmark",
        repetition=1,
        requested_output_tokens=12,
        prompt_tokens=10,
        output_tokens=12,
        generation_tokens_per_second=6.0,
        prompt_tokens_per_second=5.0,
        client_ttft_seconds=0.2,
        client_duration_seconds=2.0,
    )
    return Experiment(
        framework_version="0.1.0",
        model="sample-model",
        model_digest="synthetic",
        hardware=HardwareInfo(platform="synthetic", python_version="3", cpu="synthetic"),
        settings={},
        warmup_count=0,
        repetition_count=1,
        samples=[sample],
        summaries=summarize_samples([sample]),
        synthetic=True,
    )


def observation(t: float, value: float | None, metric: str = "power", **kwargs: Any) -> Observation:
    return Observation(
        **capability(metric, "fixture", **kwargs).model_dump(),
        invocation_id="run",
        sequence=int((t + 1) * 10),
        timestamp_utc=datetime.now(timezone.utc),
        offset_seconds=t,
        value_float=value,
    )


def fixture_experiment(telemetry: bool = True):
    experiment = make_experiment()
    experiment.settings = {
        "num_ctx": 4096,
        "num_predict": 128,
        "telemetry_interval_seconds": 1,
        "telemetry_host_role": "inference_host",
    }
    experiment.samples[0].invocation_id = "run"
    experiment.samples[0].telemetry_duration_seconds = 1
    if telemetry:
        for metric, value in (
            ("power", 40),
            ("gpu_utilization", 75),
            ("cpu_utilization", 10),
            ("ram_used", 1000000000),
            ("temperature", 65),
        ):
            points = [observation(t, value, metric) for t in [-0.1, 0.5, 1.1]]
            if metric == "ram_used":
                for point in points:
                    point.value_int, point.value_float = int(value), None
            experiment.telemetry.extend(points)
        experiment.telemetry_summaries = summarize_telemetry(experiment.telemetry, 1, 1, 12)
    return experiment
