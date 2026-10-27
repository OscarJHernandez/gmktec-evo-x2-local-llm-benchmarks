"""OllamaBackend is driven through a fake urlopen; no network is contacted."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from llm_bench.backends import ollama
from llm_bench.backends.ollama import OllamaBackend, OllamaError


class FakeResponse:
    def __init__(self, body: bytes | list[bytes]) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        assert isinstance(self.body, bytes)
        return self.body

    def __iter__(self) -> Iterator[bytes]:
        assert isinstance(self.body, list)
        return iter(self.body)


def lines(*events: dict[str, Any] | bytes) -> list[bytes]:
    return [e if isinstance(e, bytes) else json.dumps(e).encode() + b"\n" for e in events]


def patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[Any]:
    """Return a list recording each Request; outcome is a response or an exception."""
    requests: list[Any] = []

    def fake(request: Any, timeout: float | None = None) -> FakeResponse:
        requests.append(request)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome

    monkeypatch.setattr(ollama, "urlopen", fake)
    return requests


def http_error(code: int) -> HTTPError:
    return HTTPError("http://ollama.invalid", code, "error", {}, None)  # type: ignore[arg-type]


def json_response(value: Any) -> FakeResponse:
    return FakeResponse(json.dumps(value).encode())


def test_health_returns_version_and_normalizes_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = patch_urlopen(monkeypatch, json_response({"version": "0.9.0"}))
    assert OllamaBackend("http://host:1/").health() == "0.9.0"
    assert requests[0].full_url == "http://host:1/api/version"


@pytest.mark.parametrize("payload", [{}, {"version": 3}, ["0.9.0"]])
def test_health_rejects_malformed_payload(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    patch_urlopen(monkeypatch, json_response(payload))
    with pytest.raises(OllamaError, match="Malformed Ollama version"):
        OllamaBackend("http://host").health()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (http_error(404), "model not found"),
        (http_error(500), "HTTP 500"),
        (URLError("refused"), "Cannot communicate"),
        (TimeoutError("slow"), "Cannot communicate"),
    ],
)
def test_request_failures_become_ollama_errors(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException, message: str
) -> None:
    patch_urlopen(monkeypatch, failure)
    with pytest.raises(OllamaError, match=message):
        OllamaBackend("http://host").health()


@pytest.mark.parametrize("body", [b"not json", b"\xff\xfe"])
def test_request_rejects_undecodable_body(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    patch_urlopen(monkeypatch, FakeResponse(body))
    with pytest.raises(OllamaError, match="Cannot communicate"):
        OllamaBackend("http://host").health()


@pytest.mark.parametrize(
    "payload", [[], {"models": "x"}, {"models": [{"name": "a"}, "b"]}, {"other": []}]
)
def test_models_rejects_malformed_payload(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    patch_urlopen(monkeypatch, json_response(payload))
    with pytest.raises(OllamaError, match="Malformed Ollama model list"):
        OllamaBackend("http://host").models()


def test_find_model_exact_latest_alias_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = [{"name": "qwen:latest"}, {"model": "llama:8b"}, {"name": "phi:3b"}]
    patch_urlopen(monkeypatch, json_response({"models": installed}))
    backend = OllamaBackend("http://host")
    assert backend.find_model("qwen")["name"] == "qwen:latest"
    assert backend.find_model("llama:8b")["model"] == "llama:8b"
    assert backend.find_model("phi:3b")["name"] == "phi:3b"
    with pytest.raises(OllamaError, match="not installed; no model was downloaded"):
        backend.find_model("phi")


def test_residency_sanitizes_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {
        "name": "qwen:latest",
        "digest": 5,
        "size": 100,
        "size_vram": True,
        "context_length": -1,
    }
    patch_urlopen(monkeypatch, json_response({"models": [{"name": "other"}, "junk", item]}))
    assert OllamaBackend("http://host").residency("qwen") == {
        "resident": True,
        "name": "qwen:latest",
        "digest": None,
        "size": 100,
        "size_vram": None,
        "context_length": None,
    }


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (json_response({"models": []}), {"resident": False}),
        (json_response({"models": "bad"}), None),
        (json_response([]), None),
        (URLError("down"), None),
    ],
)
def test_residency_is_best_effort(
    monkeypatch: pytest.MonkeyPatch, outcome: Any, expected: Any
) -> None:
    patch_urlopen(monkeypatch, outcome)
    assert OllamaBackend("http://host").residency("qwen") == expected


def test_pull_reports_progress_and_requires_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events = lines({"status": "pulling", "completed": 1}, b"\n", {"status": "success"})
    requests = patch_urlopen(monkeypatch, FakeResponse(events))
    seen: list[dict[str, Any]] = []
    OllamaBackend("http://host").pull("qwen:7b", progress=seen.append)
    assert [event["status"] for event in seen] == ["pulling", "success"]
    assert json.loads(requests[0].data) == {"model": "qwen:7b", "stream": True}


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (lines({"status": "pulling"}), "ended without confirmation"),
        (lines({"error": "disk full"}), "failed to pull qwen: disk full"),
        (lines(b"{broken"), "Malformed Ollama model-pull"),
        (lines(b"[1]"), "Malformed Ollama model-pull"),
    ],
)
def test_pull_failures(monkeypatch: pytest.MonkeyPatch, stream: list[bytes], message: str) -> None:
    patch_urlopen(monkeypatch, FakeResponse(stream))
    with pytest.raises(OllamaError, match=message):
        OllamaBackend("http://host").pull("qwen")


@pytest.mark.parametrize(
    ("failure", "message"),
    [(http_error(500), "HTTP 500"), (URLError("reset"), "was interrupted")],
)
def test_pull_transport_errors(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException, message: str
) -> None:
    patch_urlopen(monkeypatch, failure)
    with pytest.raises(OllamaError, match=message):
        OllamaBackend("http://host").pull("qwen")


def test_generate_merges_stream_and_records_first_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = lines(
        {"response": "", "done": False},
        {"response": "Hel", "done": False},
        b"  \n",
        {"response": "lo", "done": True, "eval_count": 2},
        {"response": "ignored after done", "done": True},
    )
    requests = patch_urlopen(monkeypatch, FakeResponse(stream))
    result = OllamaBackend("http://host").generate(
        "qwen", "hi", options={"seed": 1}, think=True, keep_alive=0
    )
    assert result.payload["response"] == "Hello"
    assert result.payload["eval_count"] == 2
    assert result.client_ttft_seconds is not None
    assert 0 <= result.client_ttft_seconds <= result.client_duration_seconds
    sent = json.loads(requests[0].data)
    assert sent == {
        "model": "qwen",
        "prompt": "hi",
        "stream": True,
        "think": True,
        "options": {"seed": 1},
        "keep_alive": 0,
    }


def test_generate_without_output_has_no_ttft_and_omits_keep_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = patch_urlopen(monkeypatch, FakeResponse(lines({"response": "", "done": True})))
    result = OllamaBackend("http://host").generate("qwen", "hi", options={})
    assert result.client_ttft_seconds is None
    assert "keep_alive" not in json.loads(requests[0].data)


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        ([], "ended without a final response"),
        (lines({"response": "x", "done": False}), "ended without a final response"),
        (lines({"error": "boom"}), "returned an error response"),
        (lines(b"{broken"), "Malformed streaming response"),
        (lines(b"3"), "Malformed streaming response"),
    ],
)
def test_generate_stream_failures(
    monkeypatch: pytest.MonkeyPatch, stream: list[bytes], message: str
) -> None:
    patch_urlopen(monkeypatch, FakeResponse(stream))
    with pytest.raises(OllamaError, match=message):
        OllamaBackend("http://host").generate("qwen", "hi", options={})


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (http_error(404), "model not found"),
        (http_error(503), "HTTP 503"),
        (URLError("refused"), "Cannot communicate"),
    ],
)
def test_generate_transport_errors(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException, message: str
) -> None:
    patch_urlopen(monkeypatch, failure)
    with pytest.raises(OllamaError, match=message):
        OllamaBackend("http://host").generate("qwen", "hi", options={})
