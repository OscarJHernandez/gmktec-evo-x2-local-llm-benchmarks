"""Validated configuration and result schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean, median, stdev
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    api_url: str = "http://localhost:11434"
    repeat: int = Field(default=5, ge=1, le=100)
    warmup: int = Field(default=1, ge=0, le=20)
    num_ctx: int = Field(default=4096, ge=1)
    num_predict: int = Field(default=128, ge=1, le=32768)
    seed: int = 42
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    think: bool = False
    timeout: float = Field(default=600.0, gt=0.0)

    @field_validator("model")
    @classmethod
    def model_name_is_safe(cls, value: str) -> str:
        if any(char.isspace() for char in value) or "/" in value or "\\" in value:
            raise ValueError("model must be a local Ollama model name, not a path or URL")
        return value

    @field_validator("api_url")
    @classmethod
    def api_url_is_local_http(cls, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_url must be an http(s) URL")
        return value.rstrip("/")


class Workload(BaseModel):
    name: str
    prompt: str
    output_limit: int = Field(ge=1)


class HardwareInfo(BaseModel):
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    platform: str
    python_version: str
    cpu: str
    logical_cpu_count: int | None = None
    total_ram_bytes: int | None = None
    available_ram_bytes: int | None = None
    ram_percent: float | None = None
    gpu: str | None = None
    gpu_memory_bytes: int | None = None
    gpu_memory_source: str | None = None


class Sample(BaseModel):
    timestamp_utc: datetime | None = None
    workload: str
    phase: str
    repetition: int
    requested_output_tokens: int
    prompt_tokens: int
    output_tokens: int
    generation_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    client_ttft_seconds: float | None
    client_duration_seconds: float
    server_load_seconds: float | None = None
    server_total_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    eval_seconds: float | None = None
    prompt_cache_hit: bool | None = None
    done_reason: str | None = None
    premature_completion: bool = False
    client_metrics: bool = True
    server_metrics: bool = True


class Summary(BaseModel):
    count: int
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    samples: list[float] = Field(default_factory=list)

    @classmethod
    def from_values(cls, values: list[float]) -> Summary:
        if not values:
            return cls(count=0)
        return cls(
            count=len(values),
            mean=mean(values),
            median=median(values),
            standard_deviation=stdev(values) if len(values) > 1 else 0.0,
            samples=values,
        )


class Experiment(BaseModel):
    schema_version: str = "1.1"
    experiment_id: str = Field(default_factory=lambda: uuid4().hex)
    framework_version: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    model_digest: str | None = None
    quantization: str | None = None
    backend: str = "ollama"
    backend_version: str | None = None
    hardware: HardwareInfo
    settings: dict[str, Any]
    warmup_count: int
    repetition_count: int
    samples: list[Sample]
    summaries: dict[str, Summary]
    warnings: list[str] = Field(default_factory=list)
    synthetic: bool = False


def summarize_samples(samples: list[Sample]) -> dict[str, Summary]:
    values: dict[str, list[float]] = {}
    metrics = (
        "generation_tokens_per_second",
        "prompt_tokens_per_second",
        "client_ttft_seconds",
        "client_duration_seconds",
        "server_load_seconds",
    )
    for sample in samples:
        if sample.phase == "cold_start" and sample.server_load_seconds is not None:
            values.setdefault("cold_start_load_seconds", []).append(sample.server_load_seconds)
        if sample.phase not in {"benchmark", "warm_start"}:
            continue
        for key in metrics:
            value = getattr(sample, key)
            if value is None:
                continue
            summary_key = key
            if key == "prompt_tokens_per_second":
                cache_state = {
                    False: "uncached",
                    True: "cached",
                    None: "cache_unknown",
                }[sample.prompt_cache_hit]
                summary_key = f"{key}_{cache_state}"
            values.setdefault(summary_key, []).append(float(value))
            values.setdefault(f"workload:{sample.workload}:{summary_key}", []).append(float(value))
    return {key: Summary.from_values(items) for key, items in values.items()}
