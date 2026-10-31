---
name: write-hermetic-tests
description: Write or fix pytest tests for llm-bench that need no network, live Ollama server, GPU or real sensors. Use when adding tests, raising coverage, or debugging a flaky or environment-dependent test.
---

# Write hermetic tests

Tests must not download models, contact Ollama or the network, or need GPU hardware, and
must pass on a bare CI runner (Python 3.10-3.13).

1. Reuse builders from `tests/factories.py` (`FakeBackend`, `make_experiment`,
   `observation`, `fixture_experiment`). Put new shared helpers there; never import helpers
   from another `test_*.py`.
2. Backends: drive the code with `FakeBackend` or mocked HTTP responses. For Ollama
   behaviour see `tests/test_ollama_backend.py`.
3. Sensors: build a fake sysfs tree in `tmp_path` and pass its root to the collector
   (see `fake_sys` in `tests/test_telemetry.py`). Never read the real `/sys` or `/proc`.
4. Machine capacity: use the `benchmark_machine` fixture in `tests/conftest.py` instead of
   depending on the runner's RAM or disk; patch the probes, not the fit policy.
5. Files: write only under `tmp_path`. Locks: redirect `tempfile.gettempdir` to `tmp_path`;
   to test contention run the second acquisition via `contextvars.Context().run(...)`
   (`tests/test_storage.py::in_fresh_context`).
6. Assert behaviour, including the unhappy path: unavailable sensors stay unavailable,
   no gap filling, client and server timings stay separate, cleanup removes only
   recognised artifacts and honours `--dry-run`.
7. No sleeps, wall-clock or ordering assumptions; inject clocks and sequences.
8. Run `uv run pytest --cov` (floor 88%, branch coverage; `show_missing` lists gaps) and
   `uv run mypy` (strict, but only `llm_bench` itself). Fix coverage with tests, not by lowering
   `fail_under` in `pyproject.toml`.
