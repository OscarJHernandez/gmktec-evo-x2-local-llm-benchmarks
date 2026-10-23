"""Output-directory locks and conservative cleanup of recognized benchmark artifacts."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

from llm_bench.models import Experiment

_HELD: ContextVar[frozenset[Path]] = ContextVar("benchmark_locks", default=frozenset())
F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def output_lock(root: Path) -> Iterator[None]:
    root = root.absolute()
    canonical = root.resolve()
    if canonical in _HELD.get():
        yield
        return
    lock_root = Path(tempfile.gettempdir()) / f"llm-bench-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    if lock_root.is_symlink() or lock_root.stat().st_uid != os.getuid():
        raise ValueError("Untrusted benchmark lock directory")
    lock_name = hashlib.sha256(os.fsencode(canonical)).hexdigest()
    descriptor = os.open(lock_root / lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"Benchmark output is busy: {root}. Stop the writer before retrying."
            ) from exc
        token = _HELD.set(_HELD.get() | {canonical})
        try:
            yield
        finally:
            _HELD.reset(token)
    finally:
        os.close(descriptor)


def locked_command(function: F) -> F:
    """Hold locks across the entire CLI command, including multi-model gaps."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(function).bind(*args, **kwargs)
        roots: set[Path] = set()
        for key in ("output_dir", "input_dir"):
            path = bound.arguments.get(key)
            if isinstance(path, Path):
                roots.add(path if path.is_dir() or key == "output_dir" else path.parent)
        for key in ("output", "csv_output", "observations", "jsonl"):
            path = bound.arguments.get(key)
            if isinstance(path, Path):
                roots.add(path.parent)
        try:
            with ExitStack() as stack:
                for root in sorted(roots):
                    stack.enter_context(output_lock(root))
                return function(*args, **kwargs)
        except (OSError, ValueError) as exc:
            # Import lazily to keep persistence independent of CLI formatting.
            import typer

            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    return cast(F, wrapped)


def _known_csv(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            columns = next(csv.reader(stream), [])
        return {"model", "workload", "phase", "repetition"}.issubset(columns)
    except (OSError, UnicodeError, csv.Error):
        return False


def _known_jsonl(path: Path, catalog: bool = False) -> bool:
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                expected = (
                    {"model", "status", "candidate_index"}
                    if catalog
                    else {"experiment_id", "metric", "value"}
                )
                if not isinstance(row, dict) or not expected.issubset(row):
                    return False
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def cleanup_candidates(root: Path) -> list[Path]:
    """Return individual owned files only. Symlinks and unknown files are preserved."""
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Cleanup refuses symlinked results paths")
    if not root.exists():
        return []
    candidates: set[Path] = set()
    for path in root.glob("*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            Experiment.model_validate_json(path.read_text())
        except (ValueError, OSError):
            continue
        candidates.add(path)
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file() and not csv_path.is_symlink() and _known_csv(csv_path):
            candidates.add(csv_path)
    for path in root.glob("catalog-progress-*.jsonl"):
        if not path.is_symlink() and path.is_file() and _known_jsonl(path, catalog=True):
            candidates.add(path)
    log = root / "benchmark.log"
    if log.is_file() and not log.is_symlink():
        candidates.add(log)
    report = root / "report.html"
    if report.is_file() and not report.is_symlink():
        try:
            if "<title>Local LLM benchmark report</title>" in report.read_text():
                candidates.add(report)
        except (UnicodeError, OSError):
            pass
    exports = root / "exports"
    if exports.is_dir() and not exports.is_symlink():
        # Legacy exports at the exports root.
        for path in exports.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            if path.name.startswith("observations-") and (
                (path.suffix == ".csv" and _known_csv(path))
                or (path.suffix == ".jsonl" and _known_jsonl(path))
            ):
                candidates.add(path)
            elif path.name == "report.html":
                try:
                    if "<title>Local LLM benchmark report</title>" in path.read_text():
                        candidates.add(path)
                except (OSError, UnicodeError):
                    pass
        allowed = {
            "experiments.parquet",
            "inference_runs.parquet",
            "telemetry.parquet",
            "telemetry_summary.parquet",
            "observations.csv",
            "observations.jsonl",
            "report.html",
        }
        for snapshot in exports.iterdir():
            if snapshot.is_symlink() or not snapshot.is_dir():
                continue
            manifest_path = snapshot / "manifest.json"
            if manifest_path.is_symlink():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                files = manifest.get("files")
                if manifest.get("format") != "llm-bench-export" or not isinstance(files, list):
                    continue
                if not files or any(
                    not isinstance(name, str) or name not in allowed for name in files
                ):
                    continue
                owned = [snapshot / name for name in files]
                if any(
                    path.is_symlink() or (path.exists() and not path.is_file()) for path in owned
                ):
                    continue
                candidates.update(path for path in owned if path.is_file())
                candidates.add(manifest_path)
            except (OSError, ValueError, AttributeError):
                continue
    return sorted(candidates)


def clean_results(root: Path, *, dry_run: bool = False) -> tuple[list[Path], list[Path]]:
    with output_lock(root):
        if not root.exists():
            return [], []
        files = cleanup_candidates(root)
        if not dry_run:
            for path in sorted(files, key=lambda p: (p.name == "manifest.json", str(p))):
                if path.is_symlink():
                    raise ValueError(f"Artifact became a symlink: {path}")
                path.unlink()
            for parent in sorted({p.parent for p in files if p.parent != root}, reverse=True):
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
        preserved = [p for p in root.iterdir() if p.name != ".benchmark.lock" and p not in files]
        return files, sorted(preserved)
