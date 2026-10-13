"""Shared backend result and inference protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class GenerateResult:
    payload: dict[str, Any]
    client_duration_seconds: float
    client_ttft_seconds: float | None


class InferenceBackend(Protocol):
    def health(self) -> str: ...

    def find_model(self, name: str) -> dict[str, Any]: ...

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any],
        think: bool = False,
        keep_alive: str | int | None = None,
    ) -> GenerateResult: ...
