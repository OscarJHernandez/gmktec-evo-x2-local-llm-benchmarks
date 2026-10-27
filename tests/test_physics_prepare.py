"""Dataset preparation and run-argument validation, with urlopen replaced."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from llm_bench import physics
from llm_bench.physics import dataset_sources, prepare, run


class Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> Body:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self.data if limit < 0 else self.data[:limit]


def patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[str]:
    urls: list[str] = []

    def fake(request: Any, timeout: float | None = None) -> Body:
        urls.append(request.full_url)
        if isinstance(outcome, BaseException):
            raise outcome
        return Body(outcome)

    monkeypatch.setattr(physics, "urlopen", fake)
    return urls


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        dataset_sources("nope")


def test_prepare_downloads_json_once_and_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = patch_urlopen(monkeypatch, json.dumps([{"problem": "p"}]).encode())
    paths = prepare("phybench", tmp_path)
    assert [p.name for p in paths] == ["PHYBench-fullques_v1.json"]
    assert json.loads(paths[0].read_text()) == [{"problem": "p"}]
    assert not list(tmp_path.rglob("*.partial"))
    assert len(urls) == 1

    again = patch_urlopen(monkeypatch, URLError("must not be contacted"))
    assert prepare("phybench", tmp_path) == paths
    assert again == []


def test_prepare_validates_jsonl_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, b'{"a": 1}\n{"b": 2}\n')
    paths = prepare("ugphysics", tmp_path)
    assert paths and all(p.suffix == ".jsonl" for p in paths)


@pytest.mark.parametrize("data", [b"", b"   \n", b"{broken", b"\xff\xfe", b'{"ok": 1}\n{broken'])
def test_prepare_rejects_unexpected_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes
) -> None:
    patch_urlopen(monkeypatch, data)
    with pytest.raises(ValueError, match="Unexpected dataset format"):
        prepare("ugphysics", tmp_path)
    assert not list(tmp_path.rglob("*.jsonl"))


def test_prepare_rejects_oversized_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, b" " * (64 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds 64 MiB"):
        prepare("phybench", tmp_path)


def test_prepare_wraps_network_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, URLError("offline"))
    with pytest.raises(RuntimeError, match="Cannot retrieve phybench"):
        prepare("phybench", tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"models": []},
        {"models": ["m"], "repeat": 0},
        {"models": ["m"], "max_questions": 0},
        {"models": ["m"], "temperature": 2.5},
        {"models": ["m"], "num_ctx": 0},
        {"models": ["m"], "num_predict": 0},
    ],
)
def test_run_validates_arguments_before_touching_data(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        run(dataset="scibench", root=tmp_path, output=tmp_path / "out.jsonl", **kwargs)
    assert not (tmp_path / "out.jsonl").exists()


def test_run_requires_cached_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="just PHYSICS_DATASET=scibench physics-prepare"):
        run(dataset="scibench", models=["m"], root=tmp_path, output=tmp_path / "out.jsonl")
