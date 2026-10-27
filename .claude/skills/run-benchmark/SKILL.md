---
name: run-benchmark
description: Plan, start, monitor, stop and export an llm-bench inference benchmark sweep against Ollama. Use when the user asks to benchmark models, check progress, stop a run, or prepare data for analysis.
---

# Run a benchmark

Commands are `just` recipes (install: `uv tool install rust-just`; `tmux` is needed
for background jobs). Run `just` to list them. Settings are overridden **before** the
recipe name: `just REPEAT=5 benchmark-start` (not `just benchmark-start REPEAT=5`).

## Workflow

1. `just doctor`: checks Python, RAM, sensors and that Ollama answers.
2. `just benchmark-plan`: previews candidates and RAM/storage fit. No downloads, no
   inference. Always do this first.
3. `just benchmark-start`: runs in a tmux session. Then `just benchmark-status`
   (progress bar, active model) and `just benchmark-logs`.
4. `just benchmark-stop` ends the run. Ollama and downloaded models are untouched.
5. `just benchmark-export` writes a unique Parquet snapshot under `results/exports/`.

## Choosing what runs

- `MODELS=catalog` (default): live discovery of Ollama library variants, **downloads
  missing eligible models**, benchmarks. Can take days and use a lot of disk.
- `MODELS=installed`: only what is already downloaded, no pulls.
- `MODELS=qwen3.5:9b,qwen3.5:35b`: only those installed tags.
- Useful knobs: `REPEAT` (default 50), `WARMUP`, `NUM_CTX`, `NUM_PREDICT`, `THINK`,
  `TELEMETRY`, `TELEMETRY_INTERVAL`, `RESULTS`, `DISCOVERY=curated`.

## Guardrails

- Confirm with the user before `MODELS=catalog` with `benchmark-start`: it downloads.
- One writer per results directory; a second start is refused while the session runs.
- Cleanup is destructive: stop first, preview with `just DRY_RUN=1 benchmark-clean`,
  then `just benchmark-clean`. It removes only recognised artifacts, never models,
  datasets or physics results.
- Physics evaluations are separate (`physics-*` recipes, own tmux session).
- Never commit `results/`, logs or reports containing private prompts or machine data.

Details: `docs/model-catalog.md`, `docs/telemetry.md`, `docs/methodology.md`.
