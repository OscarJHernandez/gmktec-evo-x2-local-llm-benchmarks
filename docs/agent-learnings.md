# Agent learnings

Short, curated, non-obvious lessons for people and coding agents working in this
repo. Add or update entries with the `capture-learnings` skill
(`.claude/skills/capture-learnings/SKILL.md`). Keep this file under ~150 lines and
free of private data. Durable entries get promoted into `AGENTS.md` or a skill.

## Tooling

### just: overrides go before the recipe
- **Symptom:** `just benchmark-plan REPEAT=5` fails with "justfile does not contain
  recipe `REPEAT=5`".
- **Cause:** after the recipe name, `just` treats words as recipe arguments, not settings.
- **Do this:** `just REPEAT=5 benchmark-plan`, or `REPEAT=5 just benchmark-plan`.
  The Makefile-style `make target VAR=x` order does not carry over.
- *Added 2026-10-27*

### nbstripout rewrites cell ids unless told not to
- **Symptom:** the pre-commit hook rewrites every cell `id` in
  `notebooks/benchmark_analysis.ipynb` into a noisy diff.
- **Cause:** `nbstripout` renumbers ids by default.
- **Do this:** keep `args: [--keep-id]` in `.pre-commit-config.yaml`. Outputs must stay
  empty: `tests/test_notebook.py` fails if any code cell has outputs, so a notebook
  run in JupyterLab must be stripped before committing.
- *Added 2026-10-27*

## Tests

### Testing lock contention needs a fresh context
- **Symptom:** a test that takes `output_lock(root)` twice never sees "busy".
- **Cause:** `src/llm_bench/storage.py` makes the lock re-entrant per
  `contextvars` context, so a nested acquisition succeeds.
- **Do this:** run the second acquisition via `contextvars.Context().run(...)`
  (see `tests/test_storage.py::in_fresh_context`). Redirect `tempfile.gettempdir`
  to `tmp_path` so tests never touch the real lock directory.
- *Added 2026-10-27*

### Share builders through tests/factories.py
- **Symptom:** importing helpers from another `test_*.py` module couples tests.
- **Do this:** put shared builders (`FakeBackend`, `make_experiment`, `observation`,
  `fixture_experiment`) in `tests/factories.py` and import from there.
- *Added 2026-10-27*
