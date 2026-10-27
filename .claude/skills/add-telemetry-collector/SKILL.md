---
name: add-telemetry-collector
description: Add or change a hardware telemetry metric or sensor source (sysfs, hwmon, psutil, amd-smi) in llm-bench. Use when asked to record a new sensor, support another GPU or APU, or fix a sensor reading.
---

# Add a telemetry metric

Code lives in `src/llm_bench/telemetry/`. Principles: read-only, non-blocking,
capability-probed, and honest about gaps.

1. Register the metric in `REGISTRY` (`collectors.py`: name -> unit, value type).
2. Discover it in `Collector._discover` using `self.add(metric, source, reader, ...)`
   when present, or `self.unavailable(metric, reason, ...)` when not, so the absence
   is recorded with a reason. File sensors: `self._file(path, metric, scale, device=...)`.
3. Discovery paths are injectable (`Collector(sys=Path(...))`, `discover_gpus(sys)`);
   keep them so tests need no hardware.
4. Never substitute TDP, fill gaps, or sum overlapping power domains.
   `telemetry/aggregation.py` summaries must stay coverage-aware.
5. Sampling runs on a worker so inference never waits on a sensor
   (`telemetry/sampling.py`). Do not add blocking reads to the inference path.
6. If a new column is needed, update the Arrow schemas in `analytical.py`
   (versioned) and `docs/parquet-schema.md`; update `docs/telemetry.md`.
7. Tests: build a fake sysfs tree in `tmp_path` (see `tests/test_telemetry.py`,
   helper `write`) and use `observation()` from `tests/factories.py`.
   Never read the real `/sys` in tests.
