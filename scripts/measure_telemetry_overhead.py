"""Controlled inference substitute; reports sampler cost without requiring Ollama."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from llm_bench.backends.base import GenerateResult
from llm_bench.models import BenchmarkConfig
from llm_bench.runner import run_benchmark


class ControlledBackend:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def health(self) -> str:
        return "controlled-no-inference"

    def find_model(self, name: str) -> dict[str, Any]:
        return {"name": name}

    def generate(self, model: str, prompt: str, **kwargs: Any) -> GenerateResult:
        started = time.perf_counter()
        time.sleep(self.seconds)
        elapsed = time.perf_counter() - started
        return GenerateResult(
            {
                "prompt_eval_count": 10,
                "eval_count": 10,
                "eval_duration": int(elapsed * 1e9),
                "done": True,
            },
            elapsed,
            None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=0.2)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    for enabled in (False, True):
        started = time.perf_counter()
        result = run_benchmark(
            BenchmarkConfig(
                model="controlled",
                repeat=args.repeat,
                warmup=0,
                telemetry_enabled=enabled,
                telemetry_interval_seconds=args.interval,
            ),
            ControlledBackend(args.seconds),
        )
        print(
            json.dumps(
                {
                    "telemetry_enabled": enabled,
                    "wall_seconds": time.perf_counter() - started,
                    "invocations": len(result.samples),
                    "sampling_stats": result.sampling_stats.model_dump(),
                }
            )
        )


if __name__ == "__main__":
    main()
