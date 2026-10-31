---
name: change-schema
description: Change the result models or the exported Parquet/CSV/JSONL schema and keep every consumer in sync. Use when adding, renaming, retyping or removing a field in models.py or an analytical table column.
---

# Change the result schema

Result files are a public contract: users keep old JSON and exported snapshots.

1. Add the field to `src/llm_bench/models.py` with a default so historical JSON still
   loads. Unknown values are `None`/NULL, never `0`, `NaN` or a string. Keep client-side
   and server-reported values in separate fields.
2. Exported columns: update the explicit Arrow schema and row builders in
   `src/llm_bench/analytical.py`, and `src/llm_bench/observations.py` for telemetry rows.
   Empty tables must still carry the full schema.
3. Legacy input: `src/llm_bench/persistence.py` fills defaults for old files
   (`schema_version` falls back to `1.0`). Never invent timestamps or values for them
   (see `tests/test_analytical.py::test_legacy_missing_timestamps_not_invented`).
4. Version: if the exported shape changes, bump the schema version everywhere it appears
   (`models.py`, both places in `analytical.py`, the title of `docs/parquet-schema.md`).
   Search with `grep -rn '"1\.2"' src docs` using the current number.
5. Docs: update `docs/parquet-schema.md` (tables, types, units, NULL semantics) and any
   SQL examples; `test_documented_sql_examples_execute` runs them.
6. Consumers: check `src/llm_bench/report.py`, `src/llm_bench/hardware_report.py` and
   `notebooks/benchmark_analysis.ipynb`. The notebook must stay output-free
   (`tests/test_notebook.py`).
7. Tests: extend `tests/factories.py` builders rather than hand-building objects, and
   cover a round trip plus a legacy file. See `write-hermetic-tests`.
8. Add a `CHANGELOG.md` line under `Unreleased` and mention any migration in the PR.
