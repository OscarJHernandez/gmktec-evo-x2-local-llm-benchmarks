---
name: cut-release
description: Prepare a versioned release of llm-bench (version bump, changelog, build check, tag proposal). Use when asked to release, tag, version or publish the package.
---

# Cut a release

There is no release or publish workflow yet and no git tags. This playbook prepares the
release; **creating or pushing a tag, a GitHub release or a package upload is the
maintainer's decision: stop and ask first.**

1. Start from an up-to-date `main` with a clean tree; do the work on a release branch.
2. Pick the version (0.x while the status classifier says Alpha: breaking schema or CLI
   changes bump minor). Edit `version` in `pyproject.toml`; the distribution name is
   `local-llm-benchmarks`, the import package `llm_bench`.
3. `CHANGELOG.md`: rename `## Unreleased` to `## <version> - <YYYY-MM-DD>`, add a fresh
   empty `## Unreleased` above it, and make sure the entries describe user-visible changes
   (schema bumps, new commands, removed behaviour).
4. Full CI parity: `uv sync --locked --extra analysis`, `uv run ruff format --check .`,
   `uv run ruff check .`, `uv run mypy`, `uv run pytest --cov`, `uv build`.
5. Inspect `dist/` (gitignored): the wheel must contain `llm_bench` only, no `results/`,
   `.env` or fixtures with private data. Smoke test in a throwaway venv outside the repo:
   install the wheel and run `llm-bench --help`.
6. Open the release PR with `prepare-pr`. After it merges, propose the tag
   (`v<version>`) on the merge commit and wait for explicit approval before
   `git tag` / `git push --tags`, creating a GitHub release or uploading anywhere.
7. Publishing to PyPI would need a new workflow using trusted publishing and SHA-pinned
   actions (see `update-dependencies`); propose it as a separate PR rather than
   bundling it into a release.
