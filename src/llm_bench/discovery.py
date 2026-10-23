"""Bounded, read-only discovery of standard variants in the public Ollama library.

The library HTML is not a versioned API. Failures are reported; unknown sizes are
never treated as permission to download. Registry manifests contain no weights.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from llm_bench.model_catalog import GB, Candidate

LIBRARY = "https://ollama.com/library"
REGISTRY = "https://registry.ollama.ai/v2/library"
MAX_RESPONSE = 8 * 1024 * 1024
VARIANT = re.compile(r"(?:\d+x)?[a-z]?\d+(?:\.\d+)?[bm](?:-a\d+(?:\.\d+)?[bm])?")
NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, list[str]]] = []
        self.current: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.current = (dict(attrs).get("href") or "", [])

    def handle_data(self, data: str) -> None:
        if self.current is not None and data.strip():
            self.current[1].append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current is not None:
            self.links.append(self.current)
            self.current = None


def library_families(html: str) -> list[str]:
    parser = Links()
    parser.feed(html)
    names = set()
    for href, text in parser.links:
        name = href.removeprefix("/library/")
        if href.startswith("/library/") and NAME.fullmatch(name):
            if "embedding" not in text and "embed" not in name and "cloud" not in name:
                names.add(name)
    if not names or len(names) > 1000:
        raise ValueError("Library returned no recognized model families or exceeded its limit")
    return sorted(names)


def standard_tags(html: str, family: str) -> list[str]:
    parser = Links()
    parser.feed(html)
    prefix = f"/library/{family}:"
    tags = {href[len(prefix) :] for href, _ in parser.links if href.startswith(prefix)}
    # A standard size alias uses the publisher's default quantization. Latest is
    # a fallback only, so it cannot duplicate the standard size aliases.
    selected = sorted(tag for tag in tags if VARIANT.fullmatch(tag))
    if not selected and "latest" in tags:
        selected = ["latest"]
    if not selected and tags and all("cloud" in tag for tag in tags):
        return []  # Expected exclusion, not an incomplete-discovery warning.
    if not selected or len(selected) > 50:
        raise ValueError("No recognized standard local tags (or too many tags)")
    return selected


def manifest_size(payload: Any) -> tuple[int, str]:
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 2:
        raise ValueError("Malformed registry manifest")
    layers = payload.get("layers")
    config = payload.get("config")
    if not isinstance(layers, list) or not layers or not isinstance(config, dict):
        raise ValueError("Manifest has no local model layers")
    size = 0
    digests = []
    for layer in [config, *layers]:
        if not isinstance(layer, dict) or type(layer.get("size")) is not int or layer["size"] < 0:
            raise ValueError("Manifest has an unknown artifact size")
        size += layer["size"]
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest")
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("Manifest has an invalid model digest")
            digests.append(digest)
    if size <= 0 or not digests:
        raise ValueError("Not a downloadable local model")
    # Used to avoid aliases for identical artifacts, including template/params.
    identity = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return size, identity


def fetch_metadata(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "llm-bench-catalog/1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE + 1)
    if len(data) > MAX_RESPONSE:
        raise ValueError("Registry metadata exceeded size limit")
    return str(data.decode("utf-8"))


@dataclass
class Discovery:
    candidates: list[Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    families: int = 0


def discover_library(
    *,
    fetch: Callable[[str, float], str] = fetch_metadata,
    budget_seconds: float = 180.0,
    progress: Callable[[int, int], None] | None = None,
) -> Discovery:
    deadline = time.monotonic() + budget_seconds

    def read(url: str) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Discovery time budget exhausted")
        return fetch(url, min(10.0, remaining))

    result = Discovery()
    try:
        families = library_families(read(LIBRARY))
    except (OSError, ValueError, UnicodeError) as exc:
        result.warnings.append(f"Library discovery failed: {exc}")
        return result
    result.families = len(families)

    def discover_family(family: str) -> tuple[list[tuple[Candidate, str]], list[str]]:
        candidates: list[tuple[Candidate, str]] = []
        warnings = []
        try:
            tags = standard_tags(read(f"{LIBRARY}/{family}/tags"), family)
        except (OSError, ValueError, UnicodeError) as exc:
            return [], [f"{family}: {exc}"]
        for tag in tags:
            try:
                raw = read(f"{REGISTRY}/{family}/manifests/{tag}")
                size, identity = manifest_size(json.loads(raw))
                digest = hashlib.sha256(raw.encode()).hexdigest()
                candidates.append((Candidate(f"{family}:{tag}", size / GB, digest), identity))
            except (OSError, ValueError, UnicodeError) as exc:
                warnings.append(f"{family}:{tag}: {exc}")
        return candidates, warnings

    seen = set()
    # Four concurrent metadata readers and at most 1000 family tasks, without retries.
    # map preserves deterministic selection when aliases are duplicated.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (candidates, warnings) in enumerate(pool.map(discover_family, families), 1):
            for candidate, identity in candidates:
                if identity not in seen:
                    seen.add(identity)
                    result.candidates.append(candidate)
            result.warnings.extend(warnings)
            if progress is not None:
                progress(index, len(families))
    result.candidates.sort(key=lambda item: (item.size_gb, item.name))
    return result
