# AGENTS.md

Guidance for coding agents (and humans) working in this repository. It is a public
repo: everything committed is visible to everyone.

## What this is

`llm-bench` (package `llm_bench`, `src/` layout) benchmarks locally served LLMs
through Ollama: raw inference timings, continuous hardware telemetry, energy when
the hardware exposes it, Parquet exports, an offline HTML report and a
DuckDB/Polars notebook. Primary target: Ubuntu on a GMKtec EVO-X2 (Ryzen AI MAX+ 395).

## Commands

Python 3.10+ and [uv](https://docs.astral.sh/uv/). Never `pip install`; the lockfile
(`uv.lock`) is authoritative.

```bash
uv sync --extra analysis            # dev tools come from the default `dev` group
uv run ruff format .                # CI runs `ruff format --check .`
uv run ruff check .
uv run mypy                         # strict
uv run pytest                       # add --cov for coverage
uv build
uv run llm-bench --help
```

`just` (install with `uv tool install rust-just`) wraps these and the benchmark
workflows: `just` lists recipes, `just check` runs the CI checks. Settings are
overridden as `just REPEAT=5 benchmark-start`.

Run all of these before declaring work done. CI runs the same steps on Python
3.10-3.13.

## Layout

| Path | Purpose |
|---|---|
| `src/llm_bench/cli.py` | Typer CLI entry point (`llm-bench`) |
| `src/llm_bench/runner.py`, `workloads.py` | Benchmark orchestration and deterministic workloads |
| `src/llm_bench/backends/` | Inference engines behind the `InferenceBackend` protocol (`base.py`); only Ollama today |
| `src/llm_bench/telemetry/` | Read-only sensor discovery, collectors, sampling, aggregation |
| `src/llm_bench/models.py` | Pydantic config and result schemas |
| `src/llm_bench/model_catalog.py`, `discovery.py` | Model discovery and RAM/storage fit checks |
| `src/llm_bench/storage.py`, `progress.py` | Results-directory locking, cleanup, progress snapshots |
| `src/llm_bench/analytical.py`, `observations.py`, `persistence.py` | Parquet/CSV/JSONL exports and result I/O |
| `src/llm_bench/report.py`, `hardware_report.py` | Offline HTML report |
| `src/llm_bench/physics.py` | Opt-in physics reasoning evaluation |
| `tests/` | pytest suite (mocked sensors and Ollama) |
| `docs/` | Methodology, telemetry, schemas, catalog, analysis, backend guide |
| `notebooks/benchmark_analysis.ipynb` | Analysis notebook (committed **without outputs**) |
| `examples/` | Synthetic fixtures and a sample report |

## Rules that matter

- **Tests are hermetic.** They must not download models, contact a live Ollama
  server or the network, or need GPU hardware. Use mocked backend responses and
  injectable sysfs paths (see `tests/test_telemetry.py`, `tests/conftest.py`).
- **Never commit `results/`**, raw reports, logs, `.env`, or model files. Do not paste
  hostnames, IPs, usernames, home paths or serial numbers into code, docs or tests.
- **Notebook outputs are never committed.** `tests/test_notebook.py` fails if any
  code cell has outputs. The pre-commit `nbstripout` hook handles this.
- **Preserve measurement provenance.** Client-side and server-reported timings stay
  separate; unavailable sensor values stay unavailable (no guessing, no gap filling,
  no TDP substitution). Synthetic fixtures must be labelled synthetic.
- **No implicit downloads and no destructive actions.** Pulling weights happens only
  via the explicit `--auto-pull` catalog sweep; cleanup (`storage.py`) removes only
  recognised artifacts and supports `--dry-run`.
- Update the matching file in `docs/` and `CHANGELOG.md` (`Unreleased`) when behaviour
  changes.

## Style

`from __future__ import annotations` in every module, full type annotations (mypy
strict), ruff with 100-column lines, small focused functions, comments only for
non-obvious intent. Match the surrounding code.

## Pull requests

Keep PRs focused, use the PR template, and list how you verified the change.
