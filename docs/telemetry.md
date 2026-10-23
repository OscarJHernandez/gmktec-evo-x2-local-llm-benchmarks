# Hardware telemetry

Telemetry is enabled by default in `run`, `compare`, and `catalog-run`.
Use `--no-telemetry` or `--telemetry-interval 0.5`; Make accepts `TELEMETRY=off`
and `TELEMETRY_INTERVAL=0.5`. Intervals must be finite and positive.

## Discovery and sources

CPU identity comes from `/proc/cpuinfo`, then platform information. DRM sysfs
identifies GPUs by PCI device, deduplicates card aliases, and uses driver product
names or an installed PCI database when available. Unresolved marketing names
retain vendor/device IDs rather than an assumed Radeon model.

Collectors probe once and cache sensor paths. Sources include:

| Source | Measurements |
|---|---|
| psutil/proc | CPU/per-core utilization, RAM/swap, load, disk/network byte counters, Ollama processes |
| CPU cpufreq sysfs | Per-core current frequency |
| DRM sysfs | GPU busy percentage, separately labeled VRAM/GTT used and capacity |
| hwmon | Temperature, fans, clocks, power and energy where exposed |
| powercap | Readable package energy counters and known wraparound ranges |
| Optional SMI JSON | Missing GPU utilization and VRAM/GTT used fields in recognized layouts |
| Ollama tags and `/api/ps` | Digest, quantization, installed size, context and model residency |

Installed `amd-smi` is preferred to `rocm-smi`. The fallback has a 0.5-second
subprocess timeout and is disabled after a command failure. No SMI package is
installed automatically. Only explicitly recognized JSON fields with PCI BDF
mapping are accepted; version-specific layouts without these fields remain
unavailable. This is deliberately limited compatibility, not universal gfx1151
support. SMI is only queried when sysfs GPU capabilities are missing.

Memory bandwidth, throttling, CPU power, fan speeds, and allocation/residency
breakdowns can remain unavailable. Missing, unreadable, malformed, or unsupported
metrics have NULL values and reasons. A missing capability is emitted once per
invocation; transient read errors are retained at their timestamps. Sources that
fail during discovery generate an experiment warning.

VRAM/GTT **capacity** is an allocation pool/limit, not additional physical RAM.
VRAM used, GTT used, system RAM, and Ollama `size_vram` are different views that
may overlap on unified-memory APUs. They must not be added to derive physical
memory consumption. Model artifact bytes are file size, not resident memory.

## Sampling and overhead

One worker per experiment primes utilization counters and collects observations
before generation, periodically during generation, and after generation. It is
joined on success, exception, or interruption. No worker remains between
experiments. Boundaries bracket the call; raw offsets can be slightly negative
or exceed the call duration. UTC timestamps locate batches; monotonic offsets
support duration/energy calculations. Values in a batch are read sequentially,
not simultaneously.

Discovery and residency API requests are outside timed generation. Periodic
collection does not block the inference thread, but competes for CPU and I/O.
Boundary collection adds wall-clock overhead outside the backend's client timing.
`sampling_stats` records collection count, total/max collection duration, missed
deadlines, and collector errors. Missed ticks are skipped, not replayed. Very
short intervals can perturb performance; compare telemetry on/off on the target.

CPU utilization is interval-based since the previous counter read; its first
small baseline and post-boundary interval are not exact instantaneous values.
Ollama process CPU can exceed 100% across cores. Process identity uses PID plus
creation time. Descendants are included; RSS can double-count shared pages and
is retained per process. Disk/network metrics are host cumulative byte counters,
not rates or model-attributed traffic. Other host activity affects all measurements.

Raw samples are retained in completed experiment JSON. Memory consumption grows
with experiment length, sensors, and sampling frequency. Interrupted unfinished
experiments retain the existing persistence behavior: only previously completed
model experiments are durable. This change does not add crash-resumable journaling.

For non-loopback endpoints, telemetry is marked `client_host` and excluded from
inference-host hardware/energy comparisons. A loopback forwarding tunnel cannot
be distinguished automatically; run the benchmark on the inference host.

## Energy

Each metric/device/source is summarized independently, including boundary readings.
Statistics include valid/missing counts, min, mean, median, max, sample standard
deviation, and coverage. Means describe samples, not time-weighted utilization.

Energy counter deltas are preferred when available. A decrease is rejected unless
a known counter range permits wraparound; multiple wraps between samples cannot
be detected. Otherwise power is integrated with the trapezoidal rule. Both methods
clip linearly to invocation boundaries, using only bracketing samples with gaps
no larger than twice the configured interval. Invalid samples break coverage.

`energy_joules` requires complete invocation coverage and at least two valid
readings. `partial_energy_joules` preserves energy over covered intervals, while
full energy stays NULL. Counter/power results have `energy_method`; integration
is a derived approximation, not a direct energy measurement. Summaries retain
the original sensor kind and provenance, and the energy method identifies the
calculation. No idle subtraction, TDP substitution, or extrapolation is performed.

**AMD APU amdgpu power may include the CPU:** it is labeled `package_or_soc`, not
GPU-only. Never add CPU, GPU, and package readings without evidence of disjoint
measurement scopes. Power input/average channels also overlap. Reports keep
sources separate; the notebook chooses one series and prefers energy counters.

Joules/generated-token includes prompt processing and loading. It is NULL when
energy is unavailable or no output tokens were generated. It is not a direct
measurement of generation-only efficiency or wall-socket electricity use.

## Cleanup

`make benchmark-clean DRY_RUN=1` previews individual files; `make benchmark-clean`
removes them. The equivalent CLI is `uv run llm-bench clean --output-dir results
[--dry-run]`. Stop background work first with `make benchmark-stop`.

Cleanup recognizes validated experiment JSON and associated CSV, catalog progress
JSONL, `benchmark.log`, framework HTML reports, legacy observation exports, and
snapshot manifests under `RESULTS/exports`. It preserves unknown files, datasets,
physics outputs, notebooks, `.venv`, and Ollama models. It refuses symlinked
results paths and never follows symlinks inside results. Custom export locations
outside the selected results tree must be managed separately.

Inference, export, report, and cleanup CLI commands share nonblocking advisory
locks stored under a user-owned temporary directory, keyed by canonical output
path. Compare/catalog hold the lock across all models. Make also checks the tmux
session. Locks cover this framework's CLI writers on the same host; external
programs editing files do not honor them. Deletion failures return nonzero and
can be retried. An empty/missing directory is a successful no-op. New runs use
new IDs, but cleanup does not force a cold model load.

## Verify on metacon

```bash
uv sync --extra dev --extra analysis
make doctor
make benchmark-start MODELS=qwen3.5:9b,qwen3.5:27b,qwen3.5:35b REPEAT=5 TELEMETRY=on TELEMETRY_INTERVAL=1
make benchmark-status
make benchmark-logs
# Wait until the benchmark is stopped/completed, then:
make benchmark-export
BENCHMARK_EXPORT_DIR=/absolute/path/printed/by/export make notebook
```

In `doctor`, check CPU/GPU identity and available sensor paths. In the notebook,
inspect units, source/scope, missingness, coverage, invocation joins, and raw traces.
Confirm SoC power is not presented as GPU-only power. Compare sensors with readable
sysfs values and your installed SMI tool. If energy is absent, inspect missing-data
reasons; do not grant root access merely to populate a chart.

To assess overhead, run comparable repetitions with `TELEMETRY=off` and then
`TELEMETRY=on`, keeping model/context/workload and background activity consistent.
Inspect recorded collection cost and throughput variability before attributing
a difference to telemetry. A controlled substitute is also available:

```bash
uv run python scripts/measure_telemetry_overhead.py --repeat 1 --seconds 1 --interval 1
```

One development-host run with four one-second substitute invocations took 4.01 s
with telemetry off and 4.36 s with telemetry on; ten collections totaled 0.33 s
(maximum 0.060 s), with no missed deadlines. These are framework/sensor overhead
observations under development-host load, not real Ollama throughput or verified
AMD sensor accuracy. Boundary reads have a greater relative cost for short calls.
No fixed overhead percentage is promised.

To verify reset: preview cleanup, clean, rerun one installed model, export again,
and check that the new manifest contains only new experiment IDs. Select the new
snapshot in the notebook; a deleted snapshot produces an actionable error.

Automated validation uses mocked hardware. Passing tests does not establish real
AMD collection compatibility on metacon.

## Interface references

- [Linux amdgpu sensors and power scope](https://docs.kernel.org/gpu/amdgpu/thermal.html)
- [hwmon units and optional attributes](https://docs.kernel.org/hwmon/sysfs-interface.html)
- [Ollama model residency](https://docs.ollama.com/api/ps)
