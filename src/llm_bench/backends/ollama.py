"""Ollama HTTP API client; all timings use monotonic client-side clocks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """An Ollama request failed or returned an invalid response."""


@dataclass
class GenerateResult:
    payload: dict[str, Any]
    client_duration_seconds: float
    client_ttft_seconds: float | None


class OllamaBackend:
    def __init__(self, api_url: str, timeout: float = 600.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, *, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.api_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            message = "model not found" if exc.code == 404 else f"HTTP {exc.code}"
            raise OllamaError(f"Ollama {message}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Cannot communicate with Ollama at {self.api_url}: {exc}") from exc

    def health(self) -> str:
        value = self._request("/api/version")
        if not isinstance(value, dict) or not isinstance(value.get("version"), str):
            raise OllamaError("Malformed Ollama version response")
        return value["version"]

    def models(self) -> list[dict[str, Any]]:
        value = self._request("/api/tags")
        if not isinstance(value, dict) or not isinstance(value.get("models"), list):
            raise OllamaError("Malformed Ollama model list response")
        return value["models"]

    def find_model(self, name: str) -> dict[str, Any]:
        normalized = name.split(":")[0]
        for model in self.models():
            model_name = model.get("name") or model.get("model")
            if model_name == name or (normalized == name and model_name == f"{name}:latest"):
                return model
        raise OllamaError(f"Model {name!r} is not installed; no model was downloaded.")

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any],
        think: bool = False,
        keep_alive: str | int | None = None,
    ) -> GenerateResult:
        request_payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": think,
            "options": options,
        }
        if keep_alive is not None:
            request_payload["keep_alive"] = keep_alive
        data = json.dumps(request_payload).encode()
        request = Request(
            f"{self.api_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first_output: float | None = None
        chunks: list[dict[str, Any]] = []
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("Malformed streaming response from Ollama") from exc
                    if not isinstance(chunk, dict):
                        raise OllamaError("Malformed streaming response from Ollama")
                    chunks.append(chunk)
                    if first_output is None and chunk.get("response"):
                        first_output = time.perf_counter() - started
                    if chunk.get("error"):
                        raise OllamaError(str(chunk["error"]))
                    if chunk.get("done") is True:
                        break
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            message = "model not found" if exc.code == 404 else f"HTTP {exc.code}"
            raise OllamaError(f"Ollama {message}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise OllamaError(f"Cannot communicate with configured Ollama endpoint: {exc}") from exc
        if not chunks or chunks[-1].get("done") is not True:
            raise OllamaError("Ollama stream ended without a final response")
        merged = dict(chunks[-1])
        merged["response"] = "".join(str(chunk.get("response", "")) for chunk in chunks)
        return GenerateResult(
            payload=merged,
            client_duration_seconds=time.perf_counter() - started,
            client_ttft_seconds=first_output,
        )
