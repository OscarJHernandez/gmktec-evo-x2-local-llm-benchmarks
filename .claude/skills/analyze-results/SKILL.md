---
name: analyze-results
description: Answer questions about llm-bench results using exported Parquet snapshots, DuckDB SQL, Polars and the analysis notebook. Use when asked to compare models, inspect telemetry or energy, build charts, or interpret benchmark data.
---

# Analyze results

1. Export first: `just benchmark-export` creates an immutable snapshot directory
   under `results/exports/<id>/` with `experiments.parquet`, `inference_runs.parquet`,
   `telemetry.parquet`, `telemetry_summary.parquet`, CSV/JSONL observations,
   `manifest.json` and `report.html`.
2. Query it with DuckDB and Polars (`uv sync --extra analysis`). Schemas and column
   meanings: `docs/parquet-schema.md` (code: `SCHEMAS` in `src/llm_bench/analytical.py`).
   Tested SQL examples: `docs/analysis.md`.
3. Interactive: `BENCHMARK_EXPORT_DIR=/abs/path/to/snapshot just notebook`
   (`just notebook-serve` shares it on a trusted network; it binds all interfaces by
   default, use `JUPYTER_HOST=127.0.0.1` for local only).

## Interpret with care (see `docs/methodology.md`)

- Repeated runs measure **timing variability**, not independent samples of model quality.
- Client-side and Ollama-reported timings are separate columns; do not merge them.
- Missing sensors are unavailable, not zero. Energy exists only where counters or
  complete power coverage allow it; APU power may cover CPU and GPU together.
- Records flagged synthetic come from fixtures and say nothing about real hardware.
- Physics scores are a numeric proxy, not official benchmark scores.

## Rules

- Do not commit notebook outputs (the `nbstripout` hook and `tests/test_notebook.py`
  enforce this), exported snapshots, or `results/`.
- Do not paste hostnames, paths or machine identifiers into shared artifacts.
- Charts: label units and sample counts, show the spread (not only means).
