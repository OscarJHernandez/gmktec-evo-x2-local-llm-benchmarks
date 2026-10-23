"""Coverage-aware summaries. No TDP substitution, gap filling, or domain summation."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, stdev

from llm_bench.telemetry.schemas import Capability, Observation, TelemetrySummary


def summarize_telemetry(
    observations: list[Observation],
    duration: float,
    interval: float,
    output_tokens: int,
) -> list[TelemetrySummary]:
    groups: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for row in observations:
        groups[(row.metric, row.device, row.source)].append(row)
    summaries = []
    for rows in groups.values():
        rows.sort(key=lambda row: row.offset_seconds)
        cap = Capability.model_validate(rows[0].model_dump())
        valid = [float(row.value) for row in rows if row.value is not None]
        covered = 0.0
        partial = 0.0
        energy_metric = cap.metric in {"power", "energy"}
        for left, right in zip(rows, rows[1:], strict=False):
            if left.value is None or right.value is None:
                continue
            a, b = left.offset_seconds, right.offset_seconds
            lo, hi = max(0.0, a), min(duration, b)
            if hi <= lo or b - a > 2 * interval:
                continue
            lv, rv = float(left.value), float(right.value)
            if cap.metric == "energy":
                delta = rv - lv
                if delta < 0:
                    if (
                        cap.counter_range is None
                        or lv > cap.counter_range
                        or rv > cap.counter_range
                    ):
                        continue
                    delta += cap.counter_range
                partial += delta * (hi - lo) / (b - a)
            elif cap.metric == "power":
                lclip = lv + (rv - lv) * (lo - a) / (b - a)
                rclip = lv + (rv - lv) * (hi - a) / (b - a)
                partial += (lclip + rclip) * 0.5 * (hi - lo)
            covered += hi - lo
        fraction = min(1.0, covered / duration) if duration > 0 else 0.0
        energy = partial if energy_metric and fraction >= 1 - 1e-9 and len(valid) >= 2 else None
        summary = TelemetrySummary(
            **cap.model_dump(exclude={"available", "reason"}),
            available=bool(valid),
            reason=None if valid else rows[0].reason,
            invocation_id=rows[0].invocation_id,
            valid_count=len(valid),
            missing_count=len(rows) - len(valid),
            minimum=min(valid) if valid else None,
            mean=mean(valid) if valid else None,
            median=median(valid) if valid else None,
            maximum=max(valid) if valid else None,
            standard_deviation=stdev(valid) if len(valid) > 1 else (0.0 if valid else None),
            coverage_seconds=covered,
            coverage_fraction=fraction,
            energy_joules=energy,
            partial_energy_joules=partial if energy_metric and covered > 0 else None,
            joules_per_generated_token=energy / output_tokens
            if energy is not None and output_tokens > 0
            else None,
            energy_method=("counter_delta" if cap.metric == "energy" else "trapezoidal")
            if energy_metric
            else None,
        )
        summaries.append(summary)
    return summaries
