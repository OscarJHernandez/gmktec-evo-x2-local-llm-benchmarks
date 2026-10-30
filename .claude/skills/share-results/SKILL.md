---
name: share-results
description: Sanitise benchmark results, logs and reports before they go into an issue, PR, discussion or screenshot of this public repo. Use when asked to attach, paste, publish or commit any benchmark output.
---

# Share results safely

This is a public repository: anything posted is visible to everyone, may be indexed, and
cannot be reliably unpublished. `results/`, `*.jsonl`, `*.csv` and `*.html` are
gitignored on purpose; the only committed report is `examples/sample-report.html`.

1. Prefer a synthetic fixture. Use `examples/` or build one with `tests/factories.py`;
   synthetic data must be labelled synthetic (`Experiment.synthetic`).
2. If real numbers are needed, share a **summary** (a table of model, tokens/s, energy),
   not raw files. Export first with `just benchmark-export` and query the snapshot
   (see `analyze-results`) instead of attaching JSON, logs or a full HTML report.
3. Raw reports contain prompts, device names, kernel/driver versions and sensor labels.
   Open the file and read it before attaching; never attach a file you have not read.
4. Scrub, then search the text you are about to post for: hostnames, usernames, home
   paths, IPs and MACs, serial numbers or UUIDs, Ollama URLs, tokens, `.env` values,
   and email addresses. Example: `grep -nEi '/home/|/Users/|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|@' file`.
5. Replace network addresses in examples with documentation ranges (`192.0.2.x`) or
   `127.0.0.1`, and paths with `<results-dir>`.
6. Keep provenance honest while redacting: leave unavailable values as unavailable, keep
   client and server timings separate, and say which hardware and Ollama version the
   numbers came from (generic names, not serials).
7. Screenshots: check the window title, terminal prompt and browser tabs.
8. Never commit anything from `results/`, raw logs, `.env` or model files. Before a PR,
   run the privacy scan in `prepare-pr`.
