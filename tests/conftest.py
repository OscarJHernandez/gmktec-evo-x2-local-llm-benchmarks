"""Explicit machine capacities for isolated inference command tests."""

from __future__ import annotations

import pytest

from llm_bench import cli


@pytest.fixture
def benchmark_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model the 128 GiB target without depending on the test runner's resources.

    Individual resource-limit tests override these defaults. Patch the CLI probes,
    rather than the fit policy or psutil, to keep the real guardrails exercised.
    """
    monkeypatch.setattr(cli, "system_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(cli, "free_storage_bytes", lambda _: 1024**4)
