---
name: update-dependencies
description: Upgrade Python dependencies, GitHub Actions or pre-commit hooks in this repo and review Dependabot pull requests. Use when asked to bump, pin, audit or fix a dependency or action version.
---

# Update dependencies

`uv.lock` is authoritative and CI installs with `uv sync --locked`; never `pip install`.

1. Read the upstream changelog for every bump. Treat a new transitive dependency, a changed
   maintainer or an install-time script as a reason to stop and ask.
2. Python packages: edit the range in `pyproject.toml` only if the new version is outside
   it, then `uv lock --upgrade-package <name>` and `uv sync --extra analysis`. Keep the
   existing upper caps (`pyarrow<24`, `duckdb<2`) unless the change is about lifting them,
   and say so in the PR.
3. Dev tools live in the `dev` dependency group. The ruff version appears in `uv.lock`
   **and** as the `rev` in `.pre-commit-config.yaml`; update both together, and check the
   other hook `rev`s still exist upstream.
4. GitHub Actions in `.github/workflows/ci.yml` are pinned to full commit SHAs. Resolve
   the new tag to its SHA yourself (`gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`),
   never pin to a tag or branch. Keep `permissions: contents: read` and `runs-on: ubuntu-latest`
   (the repo is public; CI must not run on a self-hosted runner).
5. Verify as CI does: `uv sync --locked --extra analysis`, `uv run ruff format --check .`,
   `uv run ruff check .`, `uv run mypy`, `uv run pytest --cov`, `uv build`, and
   `uv run pre-commit run --all-files`. The code supports Python 3.10-3.13; do not use
   features newer than 3.10 unless the minimum changes.
6. Dependabot (`.github/dependabot.yml`, weekly for `uv` and `github-actions`): review its
   diff the same way; only merge when CI is green. Add a `CHANGELOG.md` line under
   `Unreleased` only for bumps that change behaviour or supported versions.
7. Run `capture-learnings` if a bump needed a workaround.
