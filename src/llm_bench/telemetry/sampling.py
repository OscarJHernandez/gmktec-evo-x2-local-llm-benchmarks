"""One worker per experiment; inference never waits for periodic sensor reads."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from llm_bench.telemetry.collectors import Collector
from llm_bench.telemetry.schemas import Observation, SamplingStats


@dataclass
class Window:
    invocation_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    start: float = 0.0
    end: float = 0.0
    observations: list[Observation] = field(default_factory=list)


class Sampler:
    def __init__(self, collector: Collector | None, interval: float) -> None:
        self.collector = collector
        self.interval = interval
        self.stats = SamplingStats()
        self.commands: queue.Queue[tuple[str, Window | None, threading.Event]] = queue.Queue()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> Sampler:
        if self.collector is not None:
            self.thread = threading.Thread(
                target=self._worker, name="llm-bench-telemetry", daemon=True
            )
            self.thread.start()
        return self

    def _send(self, command: str, window: Window | None) -> None:
        if self.thread is not None:
            done = threading.Event()
            self.commands.put((command, window, done))
            done.wait()

    def __exit__(self, *args: object) -> None:
        self._send("close", None)
        if self.thread is not None:
            self.thread.join()

    @contextmanager
    def invocation(self, invocation_id: str) -> Iterator[Window]:
        window = Window(invocation_id)
        self._send("begin", window)
        window.started_at = datetime.now(timezone.utc)
        window.start = time.perf_counter()
        try:
            yield window
        finally:
            window.end = time.perf_counter()
            window.ended_at = datetime.now(timezone.utc)
            self._send("end", window)
            for observation in window.observations:
                observation.offset_seconds -= window.start

    def _collect(self, window: Window) -> None:
        assert self.collector is not None
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        try:
            rows = self.collector.read()
            if not window.observations:
                rows.extend(
                    (cap, None, cap.reason)
                    for cap in self.collector.capabilities
                    if not cap.available
                )
            sequence = window.observations[-1].sequence + 1 if window.observations else 0
            for cap, value, reason in rows:
                data = cap.model_dump()
                data.update(available=value is not None, reason=reason)
                if value is not None:
                    if cap.value_type == "int":
                        data["value_int"] = int(value)
                    elif cap.value_type == "bool":
                        data["value_bool"] = bool(value)
                    else:
                        data["value_float"] = float(value)
                window.observations.append(
                    Observation(
                        **data,
                        invocation_id=window.invocation_id,
                        sequence=sequence,
                        timestamp_utc=timestamp,
                        offset_seconds=started,
                    )
                )
        except Exception as exc:  # A collector bug must never terminate the worker or inference.
            message = f"collector failed: {type(exc).__name__}"
            if message not in self.stats.errors:
                self.stats.errors.append(message)
            sequence = window.observations[-1].sequence + 1 if window.observations else 0
            for cap in self.collector.capabilities:
                window.observations.append(
                    Observation(
                        **cap.model_dump(exclude={"available", "reason"}),
                        available=False,
                        reason=message,
                        invocation_id=window.invocation_id,
                        sequence=sequence,
                        timestamp_utc=timestamp,
                        offset_seconds=started,
                    )
                )
        finally:
            elapsed = time.perf_counter() - started
            self.stats.collection_count += 1
            self.stats.collection_seconds += elapsed
            self.stats.max_collection_seconds = max(self.stats.max_collection_seconds, elapsed)

    def _worker(self) -> None:
        active: Window | None = None
        deadline = time.perf_counter() + self.interval
        while True:
            try:
                command, window, done = self.commands.get(
                    timeout=max(0.0, deadline - time.perf_counter()) if active else None
                )
            except queue.Empty:
                assert active is not None
                self._collect(active)
                deadline += self.interval
                now = time.perf_counter()
                if deadline < now:
                    missed = int((now - deadline) / self.interval) + 1
                    self.stats.missed_deadlines += missed
                    deadline += missed * self.interval
                continue
            try:
                if command == "close":
                    return
                if command == "begin":
                    active = window
                    assert self.collector is not None and active is not None
                    try:
                        self.collector.prime()
                    except Exception as exc:
                        self.stats.errors.append(f"prime failed: {type(exc).__name__}")
                    self._collect(active)
                    deadline = time.perf_counter() + self.interval
                elif command == "end" and window is not None:
                    self._collect(window)
                    active = None
            finally:
                done.set()
