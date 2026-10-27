---
name: add-physics-dataset
description: Add a new dataset to the optional physics reasoning evaluation in llm-bench, or change how answers are scored. Use when asked to support another physics or science question set.
---

# Add a physics dataset

Everything is in `src/llm_bench/physics.py`; docs in `docs/physics-benchmarks.md`.

1. `dataset_sources(dataset)`: map local filename -> official public URL, add the
   name to `DATASETS`. Downloads are explicit (`prepare`, 64 MiB cap per file, format
   validated, no overwrite of cached files); never fetch during inference.
2. `load_questions`: add a branch mapping the dataset's fields to prompt, reference,
   id and topic. Unknown schema must raise, never invent questions. Keep solutions
   out of the prompt. Identifiers must be unique and stable.
3. `score`: scoring is a **numeric proxy** (`final_numeric_answer`, 1% relative
   tolerance), parsed without `eval`. Anything without an unambiguous numeric
   reference is `unscored_reference`, not wrong.
4. The CLI (`physics-prepare`, `physics-run`, `physics-results`) validates against
   `DATASETS`; the `justfile` default `PHYSICS_DATASET` stays `scibench`.
5. Tests: tiny synthetic files in `tmp_path` plus a fake backend
   (`tests/test_physics.py`, `tests/test_physics_prepare.py`). No network.
6. Cite the dataset's license/source in the docs and say scores are not official.
