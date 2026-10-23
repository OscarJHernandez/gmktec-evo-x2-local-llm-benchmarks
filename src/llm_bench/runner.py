"""Benchmark orchestration and metric conversion."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from llm_bench import __version__
from llm_bench.backends.base import InferenceBackend
from llm_bench.backends.ollama import OllamaBackend, OllamaError
from llm_bench.hardware import GpuCollector, collect_hardware
from llm_bench.models import BenchmarkConfig, Experiment, Sample, summarize_samples
from llm_bench.telemetry.aggregation import summarize_telemetry
from llm_bench.telemetry.collectors import Collector
from llm_bench.telemetry.sampling import Sampler
from llm_bench.workloads import WORKLOADS


def _duration_ns(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value < 0
        or not math.isfinite(value)
    ):
        return None
    return float(value) / 1_000_000_000


def _number(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise OllamaError("Malformed Ollama response: expected non-negative token counts")
    return value


def run_benchmark(
    config: BenchmarkConfig,
    backend: InferenceBackend | None = None,
    *,
    gpu_collector: GpuCollector | None = None,
    progress: Callable[[int, int, str, str, int], None] | None = None,
) -> Experiment:
    client = backend or OllamaBackend(config.api_url, config.timeout)
    version = client.health()
    model_metadata = client.find_model(config.model)
    model_name = str(model_metadata.get("name") or model_metadata.get("model") or config.model)
    experiment_id = uuid4().hex
    experiment_started = datetime.now(timezone.utc)
    hardware = collect_hardware(gpu_collector)
    collector = None
    discovery_warning = None
    if config.telemetry_enabled:
        try:
            collector = Collector()
        except (OSError, RuntimeError, ValueError) as exc:
            discovery_warning = f"Telemetry discovery unavailable: {type(exc).__name__}"
    telemetry = []
    telemetry_summaries = []

    def residency(model: str) -> dict[str, Any] | None:
        method = getattr(client, "residency", None)
        try:
            value = method(model) if callable(method) else None
            return value if isinstance(value, dict) else None
        except Exception:
            return None  # Optional metadata must never abort inference.

    samples: list[Sample] = []
    warnings: list[str] = []
    total_invocations = 1 + len(WORKLOADS) * (config.warmup + config.repeat)
    with Sampler(collector, config.telemetry_interval_seconds) as sampler:
        for workload_index, workload in enumerate(WORKLOADS):
            phases = []
            if workload_index == 0:
                phases.append(("cold_start_candidate", 1))
            phases.extend((("warmup", config.warmup), ("warm_start", config.repeat)))
            for phase, count in phases:
                for repetition in range(count):
                    if progress is not None:
                        progress(
                            len(samples), total_invocations, workload.name, phase, repetition + 1
                        )
                    output_limit = min(workload.output_limit, config.num_predict)
                    invocation_id = uuid4().hex
                    before = residency(config.model)
                    with sampler.invocation(invocation_id) as window:
                        result = client.generate(
                            config.model,
                            workload.prompt,
                            options={
                                "num_ctx": config.num_ctx,
                                "num_predict": output_limit,
                                "seed": config.seed,
                                "temperature": config.temperature,
                            },
                            think=config.think,
                            keep_alive=0 if phase == "cold_start_candidate" else None,
                        )
                    after = residency(config.model)
                    payload = result.payload
                    prompt_tokens = _number(payload.get("prompt_eval_count"))
                    output_tokens = _number(payload.get("eval_count"))
                    prompt_seconds = _duration_ns(payload.get("prompt_eval_duration"))
                    eval_seconds = _duration_ns(payload.get("eval_duration"))
                    total_seconds = _duration_ns(payload.get("total_duration"))
                    load_seconds = _duration_ns(payload.get("load_duration"))
                    prompt_cache_hit: bool | None = None
                    cache_value = payload.get("prompt_cache_hit")
                    if cache_value is None:
                        cache_value = payload.get("cached")
                    if cache_value is not None:
                        if not isinstance(cache_value, bool):
                            raise OllamaError(
                                "Malformed Ollama response: prompt cache flag is not boolean"
                            )
                        prompt_cache_hit = cache_value
                    prompt_rate = (
                        prompt_tokens / prompt_seconds
                        if prompt_tokens > 0 and prompt_seconds is not None and prompt_seconds > 0
                        else None
                    )
                    generation_rate = (
                        output_tokens / eval_seconds
                        if output_tokens > 0 and eval_seconds is not None and eval_seconds > 0
                        else None
                    )
                    done_reason = payload.get("done_reason")
                    if phase == "cold_start_candidate":
                        phase = (
                            "cold_start"
                            if load_seconds is not None and load_seconds > 0
                            else "startup_warm"
                        )
                        if phase != "cold_start":
                            warnings.append(
                                "The initial sample did not report model loading; "
                                "cold-start latency "
                                "was not observed (model may already have been resident)."
                            )
                    premature = output_tokens == 0 or done_reason in {"length", "max_tokens"}
                    if premature:
                        warnings.append(
                            f"{workload.name} {phase} sample {repetition + 1} ended before "
                            "its output limit."
                        )
                    if prompt_cache_hit is not False:
                        warnings.append(
                            f"{workload.name} {phase} sample {repetition + 1} prompt cache status "
                            "was cached or unknown; exclude its prompt rate "
                            "from uncached comparisons."
                        )
                    telemetry.extend(window.observations)
                    telemetry_summaries.extend(
                        summarize_telemetry(
                            window.observations,
                            window.end - window.start,
                            config.telemetry_interval_seconds,
                            output_tokens,
                        )
                    )
                    samples.append(
                        Sample(
                            invocation_id=invocation_id,
                            started_at=window.started_at,
                            ended_at=window.ended_at,
                            telemetry_duration_seconds=window.end - window.start,
                            residency_before=before,
                            residency_after=after,
                            timestamp_utc=datetime.now(timezone.utc),
                            workload=workload.name,
                            phase=phase,
                            repetition=repetition + 1,
                            requested_output_tokens=output_limit,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            generation_tokens_per_second=generation_rate,
                            prompt_tokens_per_second=prompt_rate,
                            client_ttft_seconds=result.client_ttft_seconds,
                            client_duration_seconds=result.client_duration_seconds,
                            server_load_seconds=load_seconds,
                            server_total_seconds=total_seconds,
                            prompt_eval_seconds=prompt_seconds,
                            eval_seconds=eval_seconds,
                            prompt_cache_hit=prompt_cache_hit,
                            done_reason=done_reason if isinstance(done_reason, str) else None,
                            premature_completion=premature,
                        )
                    )
                    if progress is not None:
                        progress(
                            len(samples), total_invocations, workload.name, phase, repetition + 1
                        )
    details = model_metadata.get("details")
    quantization = details.get("quantization_level") if isinstance(details, dict) else None
    digest = model_metadata.get("digest")
    if discovery_warning:
        warnings.append(discovery_warning)
    if collector is not None and getattr(collector, "smi_disabled", None):
        warnings.append(f"Optional SMI fallback disabled: {collector.smi_disabled}")
    parameter_size = details.get("parameter_size") if isinstance(details, dict) else None
    return Experiment(
        experiment_id=experiment_id,
        timestamp_utc=experiment_started,
        telemetry=telemetry,
        telemetry_summaries=telemetry_summaries,
        capabilities=collector.capabilities if collector else [],
        devices=collector.gpus if collector else [],
        sampling_stats=sampler.stats,
        model_size_bytes=model_metadata.get("size")
        if type(model_metadata.get("size")) is int
        else None,
        parameter_size=parameter_size if isinstance(parameter_size, str) else None,
        framework_version=__version__,
        model=model_name,
        model_digest=digest if isinstance(digest, str) else None,
        quantization=quantization if isinstance(quantization, str) else None,
        backend_version=version,
        hardware=hardware,
        settings={
            "api_endpoint": "omitted for privacy",
            "telemetry_enabled": config.telemetry_enabled,
            "telemetry_interval_seconds": config.telemetry_interval_seconds,
            "telemetry_host_role": "inference_host"
            if urlparse(config.api_url).hostname in {"localhost", "127.0.0.1", "::1"}
            else "client_host",
            "cold_start_note": (
                "Positive load_duration is a legacy phase heuristic, "
                "not proof of a fully cold model."
            ),
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict,
            "seed": config.seed,
            "temperature": config.temperature,
            "think": config.think,
            "timeout_seconds": config.timeout,
            "workloads": [workload.model_dump() for workload in WORKLOADS],
            "cache_note": (
                "Ollama API does not expose prompt cache status consistently; "
                "prompt rate is only a clean uncached measure when prompt_cache_hit is false."
            ),
        },
        warmup_count=config.warmup,
        repetition_count=config.repeat,
        samples=samples,
        summaries=summarize_samples(samples),
        warnings=sorted(set(warnings)),
    )
