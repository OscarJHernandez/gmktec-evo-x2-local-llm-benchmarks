# Optional physics reasoning suite

This suite is independent from the throughput benchmark. It does **not** run when
`make benchmark-start` runs. All experiments run against **locally installed**
Ollama models. Physics dataset fetching is explicit and does not download models.

## Quick start

```bash
make physics-start
make physics-status
make physics-logs
make physics-stop
make physics-results
```

Default: **SciBench**, 50 randomly selected questions (one shared deterministic
selection across models), **3 attempts per question**, 8,192 context tokens,
1,024 maximum generated tokens, temperature 0.3. The three attempts use distinct
seeds. Switch to other datasets and specific models:

```bash
make physics-start PHYSICS_DATASET=ugphysics PHYSICS_QUESTIONS=200 PHYSICS_MODELS=qwen3.5:9b,qwen3.5:35b PHYSICS_REPEAT=5
make physics-start PHYSICS_DATASET=phybench PHYSICS_QUESTIONS=100 PHYSICS_MODELS=installed PHYSICS_THINK=on
```

Run `make physics-prepare PHYSICS_DATASET=phybench` to cache data without
starting inference. The Makefile uses `--fetch` during `physics-start` to
download a missing official source file once and reuse it later.

## Benchmarks and source provenance

| Dataset | Source | Scope | Evaluation |
| --- | --- | --- | --- |
| SciBench | [official repository](https://github.com/mandyyyyii/scibench), `dataset/original/*.json` | College numerical science; physics plus some math and chemistry | Local scalar-numeric proxy, **not** official SciBench score |
| UGPhysics | [official dataset](https://huggingface.co/datasets/UGPhysics/ugphysics), 13 English subject JSONL files | Undergraduate physics | Only unambiguous scalar-numeric answers eligible, all other answer types unscored |
| PHYBench | [official dataset](https://huggingface.co/datasets/Eureka-Lab/PHYBench), `PHYBench-fullques_v1.json` | 100 questions with reference answers | Only scalar numeric references eligible; symbolic EED **not yet implemented** |

PHYBench publishes an additional 400 questions without their reference answers,
which this initial runner intentionally excludes. The 100 full-answer records are
not always machine-scorable with this limited numeric evaluator.

**Important:** Published leaderboard numbers use the maintainers' own scoring
protocols. Our `numeric-proxy-v1` is not directly comparable. It counts only
questions with parseable numeric references in its denominator, treats invalid
numeric model outputs as incorrect, and retains **unscored** symbolic or
multi-answer references separately. Numeric correctness uses relative tolerance
1% (and absolute tolerance 1e-9). This is a *local analysis convention*, not a
claim about each benchmark's official tolerance.

The PHYBench official [EED scorer](https://github.com/phybench-official/phybench/tree/main/EED)
and UGPhysics's hybrid judge are outside this PR. No extra hosted LLM or paid
judge is silently invoked.

## Raw storage and resumption

Each model-question-attempt is flushed to
`results/physics/<dataset>-attempts.jsonl` immediately. Records include:

- Dataset name, SHA-256 of source file, benchmark problem ID, topic and answer type
- Model name/digest/quantization, inference settings and prompt version
- Original question, prompt actually sent, reference answer and raw model response
- Extracted numeric answer, correctness or unscored reason, scorer version
- Client timing, token counts and Ollama server time counters

**Reference answers and official solutions are never inserted in the model prompt.**
The raw dataset cache is kept in `results/datasets/`, ignored by git. No
datasets, model outputs, or response text should be committed to the public repo.

Runs with the same dataset checksums, sample selection, model digest, and
inference settings resume without duplicating completed attempts. The output
file may contain multiple experiment configurations; filter by `config_id`
in pandas before analyzing. Errors are written as per-attempt records and
included separately in summaries. A missing inference response is not scored
as correct.

```python
import json
import pandas as pd

path = "results/physics/scibench-attempts.jsonl"
df = pd.DataFrame(json.loads(line) for line in open(path, encoding="utf-8"))
df = df[df.config_id == df.config_id.iloc[-1]]
eligible = df[df.score_status.isin(["scored", "invalid_prediction"])]
print(eligible.groupby("model")["correct"].mean())
```

For Bayesian comparison, use a hierarchical model with question-level random
effects and model effects. Repeated attempts on the same question are **not**
independent physics problems. Latency can be analyzed jointly with correctness
but should not be confused with scoring.

## Limits

- Text-only Ollama `/api/generate`; multimodal/image-dependent problems are not
  supported. Some official problems may reference figures not provided in text.
- SciBench includes scientific questions beyond physics. This command benchmarks
  the whole selected source distribution, not a curated physics-only subset.
- Unit/scaling expressions are shown to the model, but numeric scoring is based on
  the reference dataset coefficient. Units are **not** symbolically converted.
- Symbolic-equivalence scoring, official EED, multiple-answer judging and
  formal physics difficulty scoring remain future work.
- Sequential model ordering can introduce thermal drift. The runner randomizes
  selected questions reproducibly, but does not yet counterbalance model order.
- UGPhysics has [CC BY-NC-SA 4.0](https://huggingface.co/datasets/UGPhysics/ugphysics)
  licensing. Check source licensing before redistribution or commercial use.
