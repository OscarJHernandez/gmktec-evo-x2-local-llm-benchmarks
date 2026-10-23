"""Sensor descriptors and observations shared by collectors and analytical exports."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Capability(BaseModel):
    metric: str
    unit: str
    device: str = "system"
    source: str
    scope: str = "host"
    label: str = ""
    kind: Literal["measured", "derived", "estimated"] = "measured"
    available: bool = True
    reason: str | None = None
    value_type: Literal["int", "float", "bool"] = "float"
    counter_range: float | None = None


class Observation(Capability):
    invocation_id: str
    sequence: int
    timestamp_utc: datetime
    offset_seconds: float
    value_int: int | None = None
    value_float: float | None = None
    value_bool: bool | None = None

    @property
    def value(self) -> int | float | bool | None:
        if self.value_int is not None:
            return self.value_int
        if self.value_float is not None:
            return self.value_float
        return self.value_bool


class TelemetrySummary(Capability):
    invocation_id: str
    valid_count: int
    missing_count: int
    minimum: float | None = None
    mean: float | None = None
    median: float | None = None
    maximum: float | None = None
    standard_deviation: float | None = None
    coverage_seconds: float = 0.0
    coverage_fraction: float = 0.0
    energy_joules: float | None = None
    partial_energy_joules: float | None = None
    joules_per_generated_token: float | None = None
    energy_method: str | None = None


class SamplingStats(BaseModel):
    collection_count: int = 0
    collection_seconds: float = 0.0
    max_collection_seconds: float = 0.0
    missed_deadlines: int = 0
    errors: list[str] = Field(default_factory=list)
