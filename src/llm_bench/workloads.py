"""Deterministic benchmark workload definitions."""

from llm_bench.models import Workload

WORKLOADS = (
    Workload(
        name="short-input",
        prompt="In one sentence, explain why reproducible measurements matter.",
        output_limit=128,
    ),
    Workload(
        name="long-input",
        prompt=("Summarize the following text in three concise bullet points. " * 120),
        output_limit=128,
    ),
    Workload(
        name="long-generation",
        prompt="Write a detailed, numbered guide to organizing a community garden.",
        output_limit=512,
    ),
)
