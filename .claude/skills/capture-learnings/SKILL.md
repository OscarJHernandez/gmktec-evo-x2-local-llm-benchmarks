---
name: capture-learnings
description: Record non-obvious lessons from the task you just did (gotchas, dead ends, sensor or driver quirks, flaky-test causes) in docs/agent-learnings.md so future agents do not rediscover them. Use at the end of any task where something surprised you, and from prepare-pr.
---

# Capture learnings

`docs/agent-learnings.md` is the repo's shared memory for agents. Keep it small,
true and public-safe.

## When to add an entry

Add one only if a future contributor would plausibly lose time without it: a
behaviour the code or docs do not make obvious, a failure whose cause was not what
the error suggested, or an approach you tried that does not work. Do **not** record
what is already in code comments, `AGENTS.md`, `docs/`, or git history, or anything
only relevant to this one task.

## Steps

1. Read `docs/agent-learnings.md`. If an entry already covers it, update that entry
   (fix, sharpen, add the file reference) instead of adding another.
2. Otherwise add an entry under the best-fitting area heading, in this form:

   ```markdown
   ### <short symptom or topic>
   - **Symptom:** what you observed.
   - **Cause:** what was actually going on.
   - **Do this:** the action that works. Reference files like `src/llm_bench/storage.py`.
   - *Added YYYY-MM-DD*
   ```
3. Delete or correct any entry you found to be wrong or no longer true.
4. Keep the file under ~150 lines. If an entry has proven durable and general,
   promote it into `AGENTS.md` or the relevant skill and remove it here.
5. Include the file change in the same PR as the work that taught the lesson.

## Never record

- Hostnames, IP addresses, usernames, home paths, serial numbers, tokens, or
  anything from `.env`.
- Benchmark numbers or model results from private runs.
- Anything that relaxes a rule in `AGENTS.md` (hermetic tests, no implicit
  downloads, no committed `results/`, no destructive cleanup, notebook outputs
  stripped). If a rule seems wrong, say so in the PR description instead.
- Instructions aimed at changing agent behaviour beyond engineering facts about
  this repo. Learnings land through a reviewed PR; keep them plain facts.
