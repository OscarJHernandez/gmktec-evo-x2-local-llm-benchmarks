# Analytical schema 1.2

`make benchmark-export` publishes a unique, completed snapshot under
`results/exports/<UTC>-<UUID>/`. Use **one snapshot at a time**: exports are snapshots
of the input collection, so globbing multiple snapshots can duplicate experiments.
The manifest records schema version, creation time, row counts, files, and IDs.
Existing destinations are refused. Failed writes do not publish a completed snapshot.

## Tables and joins

| Table | Key / row grain | Important columns |
|---|---|---|
| experiments | `experiment_id` | model/digest/quantization, installed artifact bytes, parameter-size label, versions, synthetic flag, timestamps, common configuration, hardware/devices/capabilities |
| inference_runs | `experiment_id, invocation_id` | workload/phase/repetition, token counts, client/server metrics, UTC boundaries, monotonic duration, residency structs |
| telemetry | IDs + `metric, device, source, sequence` | observation UTC timestamp/monotonic offset, typed value, unit, scope/label, availability/reason/kind |
| telemetry_summary | IDs + `metric, device, source` | valid/missing counts, min/mean/median/max/stddev, coverage, energy, integration method |

Join on experiment and invocation IDs. Select or aggregate a hardware series
**before** joining to per-run inference metrics; multiple sensors otherwise
multiply throughput rows. Sources/devices represent separate measurements and
must not be summed implicitly. `sequence` identifies a batch within an invocation;
multiple metrics share it. Boundary offsets can be outside the invocation.

## Types and units

All files have explicit Arrow schemas (including empty tables), ZSTD compression,
and schema-version metadata. Timestamps are `timestamp[us, tz=UTC]`, counts/bytes
are int64, measurements are float64, flags are nullable booleans. IDs are strings.
Unknown numeric values are NULL, never strings, NaN, or synthetic zeros.

Inference columns ending `_seconds` use seconds, `_tokens_per_second` use tokens/s,
and token counts are integers. Context size is requested tokens. Model size is bytes;
`parameter_size` is the backend's descriptive label, not a measured numeric count.

Telemetry has exactly one populated `value_int`, `value_float`, or `value_bool` for
valid observations. `value_type` defines the registry representation. Common units
are `percent`, `Hz`, `degC`, `rpm`, `W`, `J`, `bytes`, `bytes/s`, `threads`, and
`processes`. Capability metadata uses the same metric registry.

`available=false` accompanies NULL values and a reason. Unsupported capabilities
have one record per invocation; transient errors have ordinary batch timestamps.
`kind` describes the original reading: measured/derived/estimated. Energy columns
are derived results identified by `energy_method=counter_delta|trapezoidal`.
`energy_joules` and J/token require complete coverage; partial energy is separate.

Common configuration fields are flattened for SQL. Hardware, devices, capabilities,
workloads, residency, and sampler statistics use typed structs/lists. `settings_json`
is an additional lossless copy of arbitrary configuration; use the typed common
columns for numeric analysis. Per-run configuration is joined from experiments.
Ollama residency `size_vram` remains API-reported allocation, not proven physical
VRAM residency. Raw telemetry and provenance are preserved.

## JSON compatibility and identifiers

JSON schemas 1.0/1.1 continue loading. Existing experiment/invocation IDs are
preserved. Missing experiment IDs use UUID5 of canonical original JSON content;
missing invocation IDs use UUID5 of experiment ID plus sample position. Copies
at different paths therefore retain identity. Missing legacy timestamps remain
NULL. No historical sensors, energy, or residency are invented.

New benchmarks generate UUIDs before generation and write schema 1.2 JSON with
raw samples, telemetry, capabilities, and summaries. Completed JSON is enough to
regenerate the snapshot offline. Identical experiment IDs are deduplicated within
an export; conflicting content or duplicate/orphan invocation observations fail
explicitly. JSON publication also refuses existing destinations.

The exact machine-readable definitions are `SCHEMAS` in
`src/llm_bench/analytical.py`; metric units/types are `REGISTRY` in
`src/llm_bench/telemetry/collectors.py`.
