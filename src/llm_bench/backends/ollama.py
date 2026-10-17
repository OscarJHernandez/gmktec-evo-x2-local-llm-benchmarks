"""Ollama HTTP API client; all timings use monotonic client-side clocks."""

from __future__ import annotations

import json
import time
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_bench.backends.base import GenerateResult


class OllamaError(RuntimeError):
    """An Ollama request failed or returned an invalid response."""


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
            message = "model not found" if exc.code == 404 else f"HTTP {exc.code}"
            raise OllamaError(f"Ollama {message}") from exc
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Cannot communicate with configured Ollama endpoint.") from exc

    def health(self) -> str:
        value = self._request("/api/version")
        version = value.get("version") if isinstance(value, dict) else None
        if not isinstance(version, str):
            raise OllamaError("Malformed Ollama version response")
        return version

    def models(self) -> list[dict[str, Any]]:
        value = self._request("/api/tags")
        if not isinstance(value, dict) or not isinstance(value.get("models"), list):
            raise OllamaError("Malformed Ollama model list response")
        model_list = value["models"]
        if any(not isinstance(model, dict) for model in model_list):
            raise OllamaError("Malformed Ollama model list response")
        return cast(list[dict[str, Any]], model_list)

    def find_model(self, name: str) -> dict[str, Any]:
        normalized = name.split(":")[0]
        for model in self.models():
            model_name = model.get("name") or model.get("model")
            if model_name == name or (normalized == name and model_name == f"{name}:latest"):
                return model
        raise OllamaError(f"Model {name!r} is not installed; no model was downloaded.")


    def pull(self, model: str) -> None:
        """Download one Ollama tag, consuming progress messages without buffering them."""
        request = Request(
            f"{self.api_url}/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        success = False
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        event = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise OllamaError("Malformed Ollama model-pull response") from exc
                    if not isinstance(event, dict):
                        raise OllamaError("Malformed Ollama model-pull response")
                    if event.get("error"):
                        raise OllamaError(f"Ollama failed to pull {model}: {event['error']}")
                    if event.get("status") == "success":
                        success = True
        except HTTPError as exc:
            raise OllamaError(f"Ollama pull of {model} failed (HTTP {exc.code})") from exc
        except OSError as exc:
            raise OllamaError(f"Ollama pull of {model} was interrupted") from exc
        if not success:
            raise OllamaError(f"Ollama pull of {model} ended without confirmation")

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
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise OllamaError("Malformed streaming response from Ollama") from exc
                    if not isinstance(chunk, dict):
                        raise OllamaError("Malformed streaming response from Ollama")
                    chunks.append(chunk)
                    if first_output is None and chunk.get("response"):
                        first_output = time.perf_counter() - started
                    if chunk.get("error"):
                        raise OllamaError("Ollama returned an error response")
                    if chunk.get("done") is True:
                        break
        except HTTPError as exc:
            message = "model not found" if exc.code == 404 else f"HTTP {exc.code}"
            raise OllamaError(f"Ollama {message}") from exc
        except OSError as exc:
            raise OllamaError("Cannot communicate with configured Ollama endpoint.") from exc
        if not chunks or chunks[-1].get("done") is not True:
            raise OllamaError("Ollama stream ended without a final response")
        merged = dict(chunks[-1])
        merged["response"] = "".join(str(chunk.get("response", "")) for chunk in chunks)
        return GenerateResult(
            payload=merged,
            client_duration_seconds=time.perf_counter() - started,
            client_ttft_seconds=first_output,
        )
