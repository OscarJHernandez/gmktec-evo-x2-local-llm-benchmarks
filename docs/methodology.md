# Benchmark methodology and limitations

Three deterministic workloads cover short input, long input, and longer generation.
Settings retain prompts, seed, temperature, context/output limits, thinking mode,
model digest/quantization, backend/framework versions, and hardware metadata.

## Inference metrics

| Metric | Definition / source |
|---|---|
| Generation throughput | Ollama `eval_count / eval_duration`, tokens/s |
| Prompt throughput | Ollama `prompt_eval_count / prompt_eval_duration`, input tokens/s |
| TTFT | Monotonic client time until the first nonempty streamed output chunk, seconds |
| Client duration | Monotonic time around the streaming API request, seconds |
| Load, total, prompt/eval durations | Server nanoseconds converted to seconds |
| Token counts | Server-reported counts; unavailable/invalid rates stay NULL |

TTFT is chunk-granular, not token-ID timing. Client and server clocks measure
different scopes and remain separate. Output limits are requests; actual generated
tokens and completion reasons are retained. Zero-token or length-limited responses
carry the existing `premature_completion` flag; analysis should inspect flagged
runs rather than interpreting every flag as model failure.

## Phases and cache state

The first workload has an initial cold-start candidate, followed by configured
warm-ups and measured repetitions. Subsequent workloads have warm-ups and measured
repetitions. Every invocation, including warm-ups, receives a stable ID and telemetry.

For compatibility, positive `load_duration` labels the initial candidate
`cold_start`; otherwise it is `startup_warm`. This heuristic does **not** prove a
fully unloaded model: positive load duration can include setup overhead. The
candidate still uses existing `keep_alive=0` semantics (unload after the request,
not before it). The next call can consequently load a model too. Inspect optional
before/after residency and per-run load timings; no server-wide unload is forced.

Prompt-cache state is often absent from Ollama. Only an explicit false enters
uncached prompt-rate summaries. Cached and unknown cases remain distinct. Neither
phase nor prompt identity establishes cache status by itself.

## Reproducibility and interpretation

Keep model/digest, quantization, context, workload, temperature, thinking mode,
backend version, hardware, and phases consistent when comparing repetitions.
Default report/notebook comparisons use `warm_start` and legacy `benchmark` runs.
Warm-ups and initial candidates remain available for separate analysis.

Host utilization includes other workloads. GPU/CPU utilization and power samples
are coarse relative to token timing; a scatter plot does not establish causality.
Artifact size, parameter count labels, resident size, and physical memory are
separate concepts. Energy includes prompt processing and loading and may cover
an entire SoC. See [telemetry](telemetry.md) for integration assumptions.

Raw JSON is the durable experiment record. Parquet is recommended for analysis;
CSV and JSONL remain available. Completed model experiments survive interrupted
catalog runs, while unfinished experiments are not checkpointed. Physics evaluation
has a separate format and is documented in [its own guide](physics-benchmarks.md).

Results are local by default. Do not commit `results/` or executed notebook outputs
containing private prompts/machine data. The committed example is explicitly
synthetic and establishes no actual model or hardware performance.
