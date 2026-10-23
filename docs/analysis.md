# Analyze with DuckDB and Polars

```bash
uv sync --extra analysis
make benchmark-export
BENCHMARK_EXPORT_DIR=/absolute/path/to/printed/snapshot make notebook
# Equivalent launch:
uv run --extra analysis jupyter lab notebooks/benchmark_analysis.ipynb
```

Open `notebooks/benchmark_analysis.ipynb` and run cells in order. Edit the path and
model/workload/context filters in its first code cell. Without an explicit path,
exactly one completed snapshot must exist; otherwise the notebook lists choices.
A deleted snapshot raises a clear error. Optional tools are installed only with
the `analysis` extra; PyArrow is the benchmark's only added runtime dependency.
PyArrow provides explicit schemas and compressed Parquet without requiring pandas;
its Linux wheel was approximately 45 MiB in the development environment. Jupyter
and its larger dependency tree remain optional.

The notebook queries Parquet through an in-memory DuckDB connection, returns
filtered results with `.pl()`, and uses Polars for grouping and Plotly for charts.
It never needs pandas or a persistent database. Synthetic results require explicit
opt-in. Commit notebooks with outputs cleared.

Each analysis groups by experiment, workload, phase, and configuration. Default
comparisons exclude warm-ups/initial candidates. Hardware comparisons choose one
device/source before joining; the chosen series is printed. Missing measurements
remain NULL, and excluded counts are shown. Raw traces are loaded only for a
selected invocation/experiment. Inspect coverage before comparing energy.

## Standalone Python loading

```python
from pathlib import Path
import duckdb
import polars as pl

snapshot = Path("results/exports/SELECT_ONE_SNAPSHOT")
con = duckdb.connect(":memory:")
for name in ("experiments", "inference_runs", "telemetry", "telemetry_summary"):
    con.read_parquet(str(snapshot / f"{name}.parquet")).create_view(name)
frame = con.sql("SELECT * FROM inference_runs WHERE phase IN ('warm_start', 'benchmark')").pl()
print(
    frame.group_by("experiment_id", "workload").agg(pl.col("generation_tokens_per_second").median())
)
```

## Verified SQL examples

Run these after creating the four views above. These queries describe repetitions
within experiments; they do not pool unlike configurations or assume independent
statistical samples. NULL is retained when measurements are absent.

Throughput and latency variability, with model artifact size and context:

```sql
SELECT e.experiment_id, e.model, e.quantization, e.model_size_bytes, e.num_ctx,
       r.workload, r.phase, count(*) AS repetitions,
       avg(r.generation_tokens_per_second) AS mean_tokens_per_second,
       stddev_samp(r.generation_tokens_per_second) AS std_tokens_per_second,
       median(r.client_ttft_seconds) AS median_ttft_seconds,
       median(r.client_duration_seconds) AS median_latency_seconds
FROM experiments e JOIN inference_runs r USING (experiment_id)
WHERE r.phase IN ('warm_start', 'benchmark') AND NOT e.synthetic
GROUP BY ALL;
```

List hardware series before choosing a device/source:

```sql
SELECT DISTINCT metric, device, source, scope, unit, available, reason
FROM telemetry_summary ORDER BY metric, device, source;
```

Power and energy per invocation, preserving unsupported values and source scope:

```sql
SELECT e.model, r.invocation_id, r.workload, r.phase,
       r.generation_tokens_per_second, s.device, s.source, s.scope,
       s.mean AS power_watts, s.coverage_fraction,
       s.energy_joules, s.joules_per_generated_token
FROM inference_runs r JOIN experiments e USING (experiment_id)
LEFT JOIN telemetry_summary s
  ON s.experiment_id = r.experiment_id AND s.invocation_id = r.invocation_id
 AND s.metric = 'power'
WHERE r.phase IN ('warm_start', 'benchmark')
  AND e.telemetry_host_role = 'inference_host' AND NOT e.synthetic;
```

The final query intentionally yields one row **per sensor**, not a pooled model
average. Add explicit `device`/`source` conditions in the join before computing
per-invocation correlations. Prefer a reliable `energy` counter over integrated
power when comparing the same measurement domain; do not sum overlapping sources.
