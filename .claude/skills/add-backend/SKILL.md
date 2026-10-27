---
name: add-backend
description: Add support for a new inference engine (for example llama.cpp or vLLM) behind llm-bench's InferenceBackend protocol. Use when asked to benchmark something other than Ollama.
---

# Add an inference backend

Read `docs/adding-a-backend.md` first; it is the contract. Today only Ollama exists.

1. Implement the protocol in `src/llm_bench/backends/base.py`
   (`health`, `find_model`, `generate` returning `GenerateResult`). Model after
   `src/llm_bench/backends/ollama.py`: stdlib `urllib`, errors raised as one
   backend-specific exception, streaming parsed line by line.
2. Provenance: keep client-measured time (monotonic `time.perf_counter`) and
   server-reported counts/timings separate. Unknown metadata is unknown, never guessed.
3. No implicit downloads. Missing model means a clear error.
4. Keep engine-specific parsing inside `backends/`. `runner.py`, telemetry, exports
   and reports must stay engine-agnostic; if they need a new field, extend
   `models.py` and the Parquet schemas deliberately (`docs/parquet-schema.md`).
5. Wire a CLI option (`cli.py` builds the backend via `_backend`) without changing
   Ollama defaults.
6. Tests: mirror `tests/test_ollama_backend.py` (fake `urlopen`, no network) and
   cover success, streaming, HTTP errors, malformed payloads and timeouts.
7. Update the docs and `CHANGELOG.md`, then run `prepare-pr`.
