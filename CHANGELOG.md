# Changelog

## Unreleased

- Continuous hardware telemetry, energy measurements when supported, Parquet
  exports, an analysis notebook and safe results cleanup.
- Live Ollama library discovery with fit checks and benchmark progress/status.
- Portable JupyterLab network hosting.
- CI runs on GitHub-hosted runners across Python 3.10-3.13 with a locked
  environment and coverage reporting; `dev` tools moved to a dependency group.
- Pre-commit hooks for file hygiene and notebook output stripping.
- Added `AGENTS.md` for coding agents.
- Test suite: Ollama backend, storage/cleanup safety, CLI and physics dataset
  preparation tests; shared builders in `tests/factories.py`; coverage floor of 88%.
- `just notebook-remote`: export, then serve the notebook on localhost for an SSH tunnel.
- Agent skills in `.claude/skills/` and a curated `docs/agent-learnings.md`.
- Replaced the Makefile with a `justfile` exposing the same recipe names and
  settings; override settings as `just REPEAT=100 benchmark-start`.
- Dependabot updates are grouped per ecosystem; supported `pyarrow` range widened
  to `<26`.

## 0.1.0

- Initial MVP with Ollama streaming generation benchmarks, JSON/CSV results,
  an offline HTML report, Typer CLI, and mock-based tests.
- Records client and Ollama timings separately and keeps all run samples.
- Additional inference backends are not included.
