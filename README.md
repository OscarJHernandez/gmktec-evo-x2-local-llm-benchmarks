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
```

`uv sync` automatically creates `.venv` and installs dependencies. `uv run`
uses that environment; manual activation is unnecessary. No root privileges or
ROCm management services are required for ordinary benchmarks.

## Run and monitor

Run the automatic library sweep without listing models:

```bash
make benchmark-plan                  # preview candidates and fit checks
make benchmark-start                 # discover, download as needed, benchmark
make benchmark-status                # progress bar, completed/remaining, active model
make benchmark-logs
make benchmark-stop
```

Defaults are **50 measured repetitions per workload** and three warm-ups.
Use `REPEAT=100` for longer sampling. Status includes download progress and the
current workload/repetition; `watch -n 5 make --no-print-directory benchmark-status`
refreshes it. Fifty runs measure timing variability, not independent model-quality
samples; see [methodology](docs/methodology.md).

The sweep selects standard variants in Ollama's public open-weight library,
checks RAM/storage, and skips oversized models. It can take days and use substantial
disk space. It does not cover every community upload or license. See the
[catalog guide](docs/model-catalog.md) for scope, storage paths, and limitations.
`MODELS=installed` avoids downloads; `MODELS=qwen3.5:9b` selects an installed model;
`DISCOVERY=curated` uses the bundled catalog instead of live discovery.

Telemetry defaults to one-second sampling. Set `TELEMETRY_INTERVAL`, `TELEMETRY=off`,
`NUM_CTX`, `NUM_PREDICT`, `WARMUP`, `THINK`, and `RESULTS` as needed. Foreground
`uv run llm-bench run --model qwen3.5:9b` and `compare` never download models.

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

To host JupyterLab for another device on your trusted network, run:

```bash
make notebook-serve
# Optional overrides:
make notebook-serve JUPYTER_PORT=8890
make notebook-serve JUPYTER_HOST=127.0.0.1  # localhost access only
```

Open `http://<server-hostname>:8888/` on the other device and enter the token
printed in the server terminal. Use the server's LAN IP if its hostname does not
resolve. The default bind address is `0.0.0.0` (all interfaces); allow the selected
port through the server's firewall if needed. Use this HTTP server only on a
trusted network. It serves the repository root and initially selects the analysis
notebook. The command stays in the foreground; press Ctrl+C and confirm shutdown
when prompted. If the port is occupied, it fails instead of choosing another port.
See the [network access guide](docs/analysis.md#network-access) for details.

## Clear data and start fresh

```bash
make benchmark-stop
make benchmark-clean DRY_RUN=1
make benchmark-clean
make benchmark-start
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
