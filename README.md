# Local LLM Benchmarks

Reproducible local Ollama benchmarks with raw inference timings, continuous
hardware telemetry, energy measurements when supported, and Parquet exports.
Analyze results in the included DuckDB–Polars Jupyter notebook or an offline HTML report.

The primary target is Ubuntu on the GMKtec EVO-X2 (Ryzen AI MAX+ 395,
Radeon 8060S, 128 GB unified RAM). Sensor availability depends on the kernel,
driver, permissions, and hardware; unsupported values remain unavailable.

## Install

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and a running
[Ollama](https://ollama.com/) server. Make and tmux are needed for background jobs.

```bash
git clone https://github.com/OscarJHernandez/gmktec-evo-x2-local-llm-benchmarks.git
cd gmktec-evo-x2-local-llm-benchmarks
uv sync
make doctor
ollama pull qwen3.5:9b
```

`uv sync` automatically creates `.venv` and installs dependencies. `uv run`
uses that environment; manual activation is unnecessary. No root privileges or
ROCm management services are required for ordinary benchmarks.

## Run and monitor

A minimal foreground benchmark:

```bash
uv run llm-bench run --model qwen3.5:9b --repeat 5
```

For background comparisons of installed models:

```bash
make benchmark-start MODELS=qwen3.5:9b,qwen3.5:27b,qwen3.5:35b REPEAT=5
make benchmark-status
make benchmark-logs
make benchmark-stop
```

Telemetry defaults to one-second sampling. Override `TELEMETRY_INTERVAL=0.5`,
use `TELEMETRY=off`, or pass `--telemetry-interval` / `--no-telemetry` to the CLI.
Set `NUM_CTX`, `NUM_PREDICT`, `WARMUP`, `THINK`, and `RESULTS` as needed.

`MODELS=installed` selects installed models. The default `MODELS=catalog`
workflow can download eligible models; read the [catalog guide](docs/model-catalog.md)
before starting it. Foreground `run` and `compare` never download models.

## Export and analyze

```bash
make benchmark-export
uv sync --extra analysis
make notebook
```

Exports create a unique directory under `results/exports/` containing
`experiments.parquet`, `inference_runs.parquet`, `telemetry.parquet`,
`telemetry_summary.parquet`, CSV/JSONL observations, a manifest, and `report.html`.
Parquet is the recommended analytical format; raw experiment JSON remains available.

Choose one snapshot in `notebooks/benchmark_analysis.ipynb`, or launch with
`BENCHMARK_EXPORT_DIR=/path/to/snapshot make notebook`. The notebook uses
DuckDB SQL → Polars → Plotly. See the [analysis guide](docs/analysis.md).
Use `make benchmark-report` for a report directly from experiment JSON.

## Clear data and start fresh

```bash
make benchmark-stop
make benchmark-clean DRY_RUN=1
make benchmark-clean
make benchmark-start MODELS=qwen3.5:9b REPEAT=5
```

Cleanup removes recognized inference results, exports, reports, and logs inside
`RESULTS`. It preserves models, datasets, physics results, notebooks, and unknown
files, and refuses to run while a writer holds the results lock. New runs get new
IDs; clearing files does not unload models. See [cleanup details](docs/telemetry.md#cleanup).

## Documentation and development

- [Methodology and limitations](docs/methodology.md)
- [Telemetry, energy, cleanup, and metacon verification](docs/telemetry.md)
- [Parquet schemas](docs/parquet-schema.md) and [notebook/SQL analysis](docs/analysis.md)
- [Model catalog](docs/model-catalog.md), [physics evaluation](docs/physics-benchmarks.md)
- [Backend development](docs/adding-a-backend.md) and [contributing](CONTRIBUTING.md)

```bash
uv sync --extra dev --extra analysis
make check
uv build
```

Tests use mocked sensors and Ollama responses. Synthetic fixtures are marked as
illustrative and do not establish performance on real AMD hardware.
