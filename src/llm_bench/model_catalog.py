"""Curated Ollama model discovery and conservative machine-fit preflight.

Catalog sizes are approximate Ollama artifact sizes in decimal GB, not peak RAM.
The list is deliberately extensible and cannot cover every public Ollama tag.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

GB = 1_000_000_000
GIB = 1024**3


@dataclass(frozen=True)
class Candidate:
    name: str
    size_gb: float


# Explicit, locally runnable tags only. No :latest duplicates, cloud endpoints,
# embedding-only models, or oversized 235B/480B/671B variants.
CATALOG: tuple[Candidate, ...] = (
    Candidate("qwen3.5:0.8b", 1.3),
    Candidate("qwen3.5:2b", 3.1),
    Candidate("qwen3.5:4b", 4.0),
    Candidate("qwen3.5:9b", 7.6),
    Candidate("qwen3.5:27b", 20.0),
    Candidate("qwen3.5:35b", 22.0),
    Candidate("qwen3.5:122b", 81.0),
    Candidate("qwen3:0.6b", 0.6),
    Candidate("qwen3:1.7b", 1.5),
    Candidate("qwen3:4b", 2.6),
    Candidate("qwen3:8b", 5.3),
    Candidate("qwen3:14b", 9.5),
    Candidate("qwen3:30b", 19.5),
    Candidate("qwen3:32b", 20.5),
    Candidate("qwen3-coder:30b", 19.5),
    Candidate("qwen2.5-coder:0.5b", 0.5),
    Candidate("qwen2.5-coder:1.5b", 1.1),
    Candidate("qwen2.5-coder:3b", 2.0),
    Candidate("qwen2.5-coder:7b", 4.9),
    Candidate("qwen2.5-coder:14b", 9.3),
    Candidate("qwen2.5-coder:32b", 20.5),
    Candidate("gemma3:1b", 0.9),
    Candidate("gemma3:4b", 3.6),
    Candidate("gemma3:12b", 8.5),
    Candidate("gemma3:27b", 17.5),
    Candidate("gpt-oss:20b", 14.5),
    Candidate("gpt-oss:120b", 66.0),
    Candidate("deepseek-r1:1.5b", 1.2),
    Candidate("deepseek-r1:7b", 4.9),
    Candidate("deepseek-r1:8b", 5.4),
    Candidate("deepseek-r1:14b", 9.3),
    Candidate("deepseek-r1:32b", 20.5),
    Candidate("deepseek-r1:70b", 44.0),
    Candidate("llama3.2:1b", 1.4),
    Candidate("llama3.2:3b", 2.1),
    Candidate("llama3.3:70b", 44.0),
    Candidate("mistral-small3.2:24b", 15.5),
    Candidate("mistral-nemo:12b", 7.4),
    Candidate("phi4:14b", 9.5),
    Candidate("command-r:35b", 20.0),
    Candidate("olmo2:7b", 4.7),
    Candidate("olmo2:13b", 8.7),
    Candidate("granite3.3:2b", 1.6),
    Candidate("granite3.3:8b", 5.1),
)


def catalog_candidates(extra: Path | None = None) -> list[Candidate]:
    """Merge extra JSON candidates without duplicating tags or trusting malformed sizes."""
    candidates = list(CATALOG)
    if extra is not None:
        data = json.loads(extra.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Catalog file must be a JSON list of {name, size_gb} entries")
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Catalog entry must be an object")
            name, size = item.get("name"), item.get("size_gb")
            if (
                not isinstance(name, str)
                or not name
                or name.endswith("-cloud")
                or "-cloud:" in name
                or "/" in name
                or "\\" in name
                or any(char.isspace() for char in name)
                or isinstance(size, bool)
                or not isinstance(size, int | float)
                or not (0 < size < 10000)
            ):
                raise ValueError(f"Invalid model catalog entry: {item!r}")
            candidates.append(Candidate(name=name, size_gb=float(size)))
    return list({item.name: item for item in candidates}.values())


def model_storage_dir(explicit: Path | None = None) -> Path:
    return explicit or Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser()


def free_storage_bytes(model_dir: Path) -> int:
    """Probe the existing parent when Ollama has not created its model directory."""
    target = model_dir.resolve()
    while not target.exists():
        if target.parent == target:
            raise OSError("Could not locate an existing model-storage ancestor")
        target = target.parent
    return shutil.disk_usage(target).free


def fit_reason(
    candidate: Candidate,
    *,
    total_ram_bytes: int,
    free_disk_bytes: int,
    already_installed: bool = False,
    headroom_gib: int = 24,
    disk_reserve_gib: int = 20,
) -> str | None:
    """Return a skip reason or None. Weight size is an estimate, not proof of fit."""
    # Budget ~30% above weights for runtime allocations and a fixed OS/KV margin.
    runtime_bytes = int(candidate.size_gb * GB * 1.3) + headroom_gib * GIB
    if runtime_bytes > total_ram_bytes:
        return "estimated runtime memory exceeds physical RAM safety budget"
    if not already_installed:
        download_bytes = int(candidate.size_gb * GB * 1.15) + disk_reserve_gib * GIB
        if download_bytes > free_disk_bytes:
            return "estimated download exceeds free storage safety budget"
    return None


def system_ram_bytes() -> int:
    return int(psutil.virtual_memory().total)


def select_installed(installed: list[dict[str, Any]]) -> list[Candidate]:
    """Include all installed generative models, excluding embedding-only and cloud."""
    result: list[Candidate] = []
    for item in installed:
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or "cloud" in name.lower():
            continue
        details = item.get("details") or {}
        families = details.get("families") or []
        if (
            "embed" in name.lower()
            or "embedding" in name.lower()
            or any("bert" in str(family).lower() for family in families)
        ):
            continue
        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            # Without model size, fail closed instead of assuming a small model.
            continue
        result.append(Candidate(name=name, size_gb=size / GB))
    return result
