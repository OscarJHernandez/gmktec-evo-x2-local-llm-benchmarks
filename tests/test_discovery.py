"""Registry fixtures exercise selection without network or weight downloads."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_bench.discovery import (
    LIBRARY,
    REGISTRY,
    discover_library,
    library_families,
    manifest_size,
    standard_tags,
)


def manifest(size: Any = 5_000_000_000, digest: str = "a") -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "config": {"size": 100},
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:" + digest * 64,
                "size": size,
            }
        ],
    }


def test_library_filters_capabilities_and_untrusted_paths() -> None:
    html = """<a href="/library/alpha">Alpha <span>tools</span></a>
    <a href="/library/embedding-model">embeddings</a>
    <a href="/library/bert">Bert<span>embedding</span></a>
    <a href="/library/alpha:latest">tag</a><a href="https://elsewhere/evil">bad</a>
    <a href="/library/../../evil">bad</a><a href="/library/alpha">duplicate</a>"""
    assert library_families(html) == ["alpha"]
    with pytest.raises(ValueError, match="no recognized"):
        library_families("<html>format changed</html>")


def test_standard_variants_avoid_quantizations_aliases_and_cloud() -> None:
    html = "".join(
        f'<a href="/library/alpha:{tag}">{tag}</a>'
        for tag in [
            "latest",
            "8b",
            "8b-q4_K_M",
            "8b-fp16",
            "70b",
            "8b-cloud",
            "e2b",
            "30b-a3b",
            "8x7b",
            "8x22b",
        ]
    )
    assert standard_tags(html, "alpha") == ["30b-a3b", "70b", "8b", "8x22b", "8x7b", "e2b"]
    assert standard_tags('<a href="/library/alpha:latest">a</a>', "alpha") == ["latest"]
    assert standard_tags('<a href="/library/alpha:8b-cloud">a</a>', "alpha") == []


@pytest.mark.parametrize("size", [None, True, -1, "100", float("nan")])
def test_unknown_manifest_size_fails_closed(size: Any) -> None:
    with pytest.raises(ValueError):
        manifest_size(manifest(size))


def test_manifest_requires_local_model_layers() -> None:
    assert manifest_size(manifest())[0] == 5_000_000_100
    with pytest.raises(ValueError, match="local model"):
        manifest_size({"schemaVersion": 2, "config": {}, "layers": []})
    value = manifest()
    value["layers"][0]["digest"] = "unknown"
    with pytest.raises(ValueError, match="digest"):
        manifest_size(value)


def test_discovery_deduplicates_manifests_and_reports_partial_failures() -> None:
    responses = {
        LIBRARY: '<a href="/library/alpha">A</a><a href="/library/beta">B</a>',
        f"{LIBRARY}/alpha/tags": (
            '<a href="/library/alpha:1b">A</a><a href="/library/alpha:2b">B</a>'
        ),
        f"{LIBRARY}/beta/tags": '<a href="/library/beta:latest">B</a>',
        f"{REGISTRY}/alpha/manifests/1b": json.dumps(manifest()),
        f"{REGISTRY}/alpha/manifests/2b": json.dumps(manifest()),
    }

    def fetch(url: str, timeout: float) -> str:
        assert 0 < timeout <= 10
        if url not in responses:
            raise OSError("offline")
        return responses[url]

    events = []
    result = discover_library(
        fetch=fetch, progress=lambda done, total: events.append((done, total))
    )
    assert [c.name for c in result.candidates] == ["alpha:1b"]
    assert result.candidates[0].size_gb == 5.0000001
    assert result.candidates[0].digest is not None
    assert result.families == 2
    assert len(result.warnings) == 1 and "beta:latest" in result.warnings[0]
    assert events == [(1, 2), (2, 2)]
    expired = discover_library(fetch=fetch, budget_seconds=0)
    assert not expired.candidates and "budget" in expired.warnings[0]


def test_discovery_failure_does_not_silently_use_stale_catalog() -> None:
    result = discover_library(fetch=lambda *_: "malformed html")
    assert not result.candidates
    assert result.warnings


def test_discovery_request_timeout_and_expected_cloud_exclusion() -> None:
    def fetch(url: str, timeout: float) -> str:
        if url == LIBRARY:
            return '<a href="/library/alpha">A</a><a href="/library/beta">B</a>'
        if url == f"{LIBRARY}/alpha/tags":
            return '<a href="/library/alpha:1b-cloud">cloud</a>'
        raise TimeoutError("timed out")

    result = discover_library(fetch=fetch)
    assert not result.candidates
    assert result.warnings == ["beta: timed out"]
