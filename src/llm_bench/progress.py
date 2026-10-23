"""Atomic progress snapshots, readable without taking the benchmark writer lock."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal
from uuid import uuid4

import psutil
from pydantic import BaseModel, Field

from llm_bench.model_catalog import Candidate


class PlannedModel(BaseModel):
    name: str
    size_gb: float | None = None
    digest: str | None = None


class ProgressState(BaseModel):
    format: Literal["llm-bench-progress-v1"]
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pid: int
    process_created: float
    stage: str = "starting"
    finished: bool = False
    models: list[PlannedModel] = Field(default_factory=list)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    current_model: str | None = None
    invocation_done: int = 0
    invocation_total: int = 0
    workload: str | None = None
    phase: str | None = None
    repetition: int = 0
    repeat: int
    warmup: int
    download_done: int | None = None
    download_total: int | None = None
    detail: str | None = None
    discovery_source: str | None = None
    discovery_warnings: list[str] = Field(default_factory=list)


class Progress:
    def __init__(self, root: Path, repeat: int, warmup: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.state = ProgressState(
            format="llm-bench-progress-v1",
            pid=os.getpid(),
            process_created=psutil.Process().create_time(),
            repeat=repeat,
            warmup=warmup,
        )
        self.path = root / (
            f"benchmark-state-{self.state.started_at:%Y%m%dT%H%M%S%fZ}-{self.state.run_id}.jsonl"
        )
        self._last_download_update = 0.0
        self._last_download_key: tuple[str, str] | None = None
        self.update()

    def __enter__(self) -> Progress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stage = (
            "finished"
            if exc_type is None
            else "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        self.update(stage=stage, finished=True)

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(self.state, key, value)
        self.state.updated_at = datetime.now(timezone.utc)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(self.state.model_dump_json() + "\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def plan(self, candidates: list[Candidate]) -> None:
        self.update(
            models=[
                PlannedModel(name=c.name, size_gb=c.size_gb, digest=c.digest) for c in candidates
            ]
        )

    def start_model(self, name: str) -> None:
        self._last_download_update = 0.0
        self._last_download_key = None
        self.update(
            current_model=name,
            stage="checking",
            invocation_done=0,
            invocation_total=0,
            workload=None,
            phase=None,
            repetition=0,
            download_done=None,
            download_total=None,
            detail=None,
        )

    def terminal(self, status: str, reason: str | None = None) -> None:
        field = (
            "completed"
            if status == "completed"
            else "skipped"
            if status.startswith("skipped")
            else "failed"
        )
        self.update(**{field: getattr(self.state, field) + 1}, stage=status, detail=reason)

    def invocation(self, done: int, total: int, workload: str, phase: str, repetition: int) -> None:
        self.update(
            stage="benchmarking",
            invocation_done=done,
            invocation_total=total,
            workload=workload,
            phase=phase,
            repetition=repetition,
        )

    def download(self, event: dict[str, Any]) -> None:
        now = time.monotonic()
        done, total = event.get("completed"), event.get("total")
        key = (str(event.get("status", "pulling")), str(event.get("digest", "")))
        final = event.get("status") == "success" or (type(total) is int and done == total)
        first_bytes = self.state.download_total is None and type(total) is int and total > 0
        if (
            now - self._last_download_update < 0.25
            and key == self._last_download_key
            and not final
            and not first_bytes
        ):
            return
        self._last_download_update = now
        self._last_download_key = key
        # Ollama reports the active layer, not aggregate model download bytes.
        self.update(
            stage="downloading",
            detail=str(event.get("status", "pulling")),
            download_done=done if type(done) is int and done >= 0 else None,
            download_total=total if type(total) is int and total > 0 else None,
        )


def latest_progress(root: Path) -> ProgressState | None:
    for path in sorted(root.glob("benchmark-state-*.jsonl"), reverse=True):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            return ProgressState.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def process_alive(state: ProgressState) -> bool:
    try:
        process = psutil.Process(state.pid)
        return bool(
            process.create_time() == state.process_created
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.Error:
        return False


def bar(done: int, total: int) -> str:
    fraction = min(1.0, max(0.0, done / total)) if total else 0.0
    filled = int(fraction * 24)
    return f"[{'#' * filled}{'-' * (24 - filled)}] {fraction:.0%}"


def status_text(root: Path) -> str:
    state = latest_progress(root)
    if state is None:
        return (
            "No benchmark progress recorded. Check benchmark.log for legacy runs or startup errors."
        )
    alive = not state.finished and process_alive(state)
    condition = (
        "RUNNING"
        if alive
        else "FINISHED"
        if state.stage == "finished"
        else "FAILED"
        if state.finished and state.stage == "failed"
        else "STOPPED / INTERRUPTED"
    )
    processed = state.completed + state.failed + state.skipped
    total = len(state.models)
    lines = [
        f"{condition}: {state.stage} | run {state.run_id}",
        f"Measured repetitions: {state.repeat} per workload; warm-ups: {state.warmup}",
    ]
    if state.stage in {"starting", "discovering"}:
        lines.append("Model total and remaining: pending discovery")
    else:
        lines.extend(
            [
                f"Models processed {bar(processed, total)} {processed}/{total}",
                f"Benchmarked: {state.completed} | Failed: {state.failed} "
                f"| Skipped: {state.skipped} "
                f"| Remaining: {max(0, total - processed)} (includes active model)",
            ]
        )
    if state.current_model:
        lines.append(f"Model: {state.current_model}")
    if state.stage == "downloading" and state.download_total:
        lines.append(
            f"Download (active layer): {bar(state.download_done or 0, state.download_total)} "
            f"{state.download_done or 0:,}/{state.download_total:,} bytes"
        )
    if state.invocation_total:
        lines.append(
            f"Invocations {bar(state.invocation_done, state.invocation_total)} "
            f"{state.invocation_done}/{state.invocation_total} (includes warm-ups/startup)"
        )
        lines.append(
            f"Workload: {state.workload} | Phase: {state.phase} | Repetition: {state.repetition}"
        )
    if state.detail:
        lines.append(state.detail)
    if state.discovery_warnings:
        lines.append(
            f"Discovery incomplete: {len(state.discovery_warnings)} warning(s); "
            "see benchmark.log and the progress snapshot."
        )
    lines.append(f"Last update: {state.updated_at.isoformat()}")
    return "\n".join(lines)
