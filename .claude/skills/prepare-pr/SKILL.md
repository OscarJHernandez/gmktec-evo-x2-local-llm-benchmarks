---
name: prepare-pr
description: Run the checks and hygiene steps needed before opening a pull request in this repo. Use before committing, pushing or opening any PR.
---

# Prepare a PR

This is a public repository.

1. `just check` (ruff format check, ruff check, mypy strict, pytest). CI also runs
   `uv run pytest --cov` with an enforced floor, and `uv build`.
2. `uv run pre-commit run --all-files` (formatting, file hygiene, `nbstripout`).
3. Behaviour changed? Update the matching `docs/*.md` and add a line under
   `Unreleased` in `CHANGELOG.md`.
4. Privacy scan of `git diff origin/main...`: no `results/`, logs, reports, notebook
   outputs, `.env`, hostnames, IPs, usernames, home paths, serials, tokens or model
   files. Fixtures must be labelled synthetic.
5. Tests added for new behaviour and still hermetic (no network, live Ollama, GPU).
6. Run `capture-learnings` if anything non-obvious came up; include that file change.
7. Use the PR template (`.github/PULL_REQUEST_TEMPLATE.md`): summary, how verified,
   limitations. Keep the PR focused on one change.
