# Adding an inference backend

The initial runner intentionally supports Ollama only. A future backend should
provide the small surface the orchestration needs:

1. A health/version check that reports actionable connection errors.
2. Installed-model metadata including immutable digest and quantization when the
   engine exposes them.
3. Generation accepting model, fixed prompt, output/context settings, seed and
   reasoning mode where supported.
4. Raw server timing/count fields and a monotonic client duration plus
   first-output timing, clearly distinguished by provenance.
5. Explicit errors for missing models, malformed responses and timeouts.

Keep engine-specific parsing in `src/llm_bench/backends/`. Do not download
weights implicitly. Represent unavailable metadata as unknown rather than
guessing; preserve raw samples and backend version. Add unit tests with mocked
responses and keep hardware and report generation independent of the engine.
