# Automatic Ollama sweeps

`make benchmark-start` uses **live library discovery**, pulls missing eligible
models, and benchmarks them sequentially in a detached tmux session. It needs no
model list. The default is **50 measured repetitions per workload**, three
warm-ups per workload, and one initial startup observation: 160 generation calls
per model across the three workloads. Use `REPEAT=100` for longer runs; supported
values are 1–1000. Five repetitions are useful for smoke tests, not the default.

## Preview, run, and monitor

```bash
make benchmark-plan                  # metadata/fit preview; no downloads or inference
make benchmark-start                 # automatic library sweep, downloads as needed
make benchmark-status                # one progress snapshot
watch -n 5 make --no-print-directory benchmark-status
make benchmark-logs
make benchmark-stop
make benchmark-export
```

`benchmark-status` shows the latest run ID, process status, model progress bar,
successfully benchmarked / failed / skipped counts, remaining models (including
the active one), current model, active-layer download bytes, and invocation
progress with workload, phase, and repetition. The model bar measures **processed
candidates**; a skipped model has not been benchmarked. During discovery the total
is not yet known, and status shows library-family discovery progress instead.
No ETA is inferred from heterogeneous model speeds.

The tmux session state and benchmark process state are reported separately. A
process identity includes PID and creation time, so a stopped run does not become
"running" when its PID is reused. Progress is saved atomically to
`RESULTS/benchmark-state-*.jsonl`; terminal model outcomes remain in
`catalog-progress-*.jsonl`. These are owned cleanup artifacts. Each new sweep has
a new ID and counts, even if older experiment results are retained. These files
track progress; they do **not** resume an unfinished model. Completed experiments
survive interruption. Explicit-model and installed-only runs also report progress.

Equivalent foreground commands:

```bash
uv run llm-bench catalog-run --auto-pull --dry-run
uv run llm-bench catalog-run --auto-pull
uv run llm-bench status --output-dir results
```

## What "all models" covers

Discovery reads the [public Ollama library](https://ollama.com/library), its family
tag pages, and registry manifests, without downloading weights. It selects each
family's standard size aliases (for example `8b`, `70b`, `8x7b`, `e2b`, or
`30b-a3b`), using
`latest` only when a family has no such aliases. These aliases select the
publisher's default quantization. It excludes embedding families and cloud tags,
and requires local weight layers with known sizes. Identical manifests are
deduplicated. Installed aliases of a discovered digest are reused without pulling
the same artifact under another name. Existing installed tags are reused without
automatically updating their weights. Additional installed generative models are
included. Candidates run in increasing artifact-size order.

This is a broad **open-weight library sweep**, not an enumeration of every
community upload, historical revision, or quantization. Licenses vary; discovery
is not an open-source license certification. Multimodal models receive the same
text workloads, not image/audio tests. Some text-capable or specialist models may
still reject these prompts or require a newer Ollama release; failures are logged
and the sweep continues.

Library HTML is not a versioned API. Discovery uses four metadata workers,
10-second request timeouts, an approximately three-minute discovery budget,
response/family/tag limits, and explicit warnings for missing or malformed data.
Partial discovery remains usable, with warnings visible in status and saved with
the frozen candidate list and manifest digests. Unknown sizes never authorize a
download. Discovery failure does not silently claim a complete sweep or fall back
to approximate catalog sizes. Installed models can still run.

For a fixed candidate list or unavailable library metadata, choose the bundled 44-tag catalog:

```bash
make benchmark-start DISCOVERY=curated
make benchmark-start MODELS=installed       # no discovery network calls or pulls
make benchmark-start MODELS=qwen3.5:9b      # named installed models only
make benchmark-plan MODELS=qwen3.5:9b       # preview that same named selection
```

## Fit and storage

Preflight budgets the complete weight artifact, **30% runtime overhead plus
24 GiB RAM reserve**, and **15% download overhead plus 20 GiB disk reserve**. It
checks physical RAM and current free storage before every pull; after pulling it
checks actual installed size again before inference. Installed models do not need
download space. Size estimates cannot guarantee fit at every context length or
under competing workloads. `NUM_CTX=4096` is the default; large contexts can need
substantially more memory. This is not a GPU-only residency guarantee.

Set `MODEL_DIR=/actual/ollama/model/storage` (or `OLLAMA_MODELS`) to the server's
weight directory, especially when Ollama runs as a system service. Without either,
the existing default is `~/.ollama/models`. The selected path is printed in the
plan and run log; it must refer to the server's filesystem for disk checks to be
meaningful. The endpoint must be loopback for automatic sweeps.

A broad sweep can take days and accumulate hundreds of GB or more. Models remain
installed for reuse; the benchmark does not delete weights to make room. When
space is insufficient, remaining downloads are skipped individually with reasons.
The preview's eligible count is based on **current** free space; it is not a promise
that all candidates can be retained together. Existing model weights, completed
results, and physics workflows are preserved.

Installed community namespaces such as `user/model:8b` are supported.

Extra candidates can be added using `CATALOG_FILE=/path/to/models.json`:

```json
[{"name": "my-model:7b", "size_gb": 5.0}]
```

This augments live discovery (or the bundled catalog with `DISCOVERY=curated`).
User-supplied sizes are estimates. `run` and `compare` never download models.

## Sampling and fresh starts

Fifty repetitions improve estimates of local timing variability over a five-run
smoke test, but no fixed count guarantees statistical precision. Repeated prompts
at temperature zero are not independent draws of model quality. Cache, thermal
drift, and run order matter; compare matched workload/context/quantization/phase
groups, inspect distributions, and repeat whole sessions for between-session
variation. See [methodology](methodology.md#repetitions-and-statistical-interpretation).

```bash
make benchmark-stop
make benchmark-clean DRY_RUN=1
make benchmark-clean
make benchmark-start
make benchmark-status
make benchmark-export
make notebook
```

Use the same `RESULTS=/path/to/results` on each command for custom directories.
Cleanup resets results and progress, preserving downloaded model weights.
