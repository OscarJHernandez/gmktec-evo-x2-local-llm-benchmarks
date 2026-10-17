# Ollama model catalog and fit checks

This suite targets locally hosted Ollama inference on the GMKtec EVO-X2 with
128 GB unified memory. The curated list in
`src/llm_bench/model_catalog.py` deliberately includes many text-capable
and multimodal models which can accept text prompts. It excludes cloud-only,
embedding-only and clearly oversized tags by default. Locally installed extra
generative model tags are discovered dynamically via Ollama `/api/tags`.

## Verified Ollama library families

The default catalog contains 44 specific model tags sourced from these official
Ollama library pages (as of October 2026):

- [Qwen3.5](https://ollama.com/library/qwen3.5/tags): 0.8B–122B
- [Qwen3](https://ollama.com/library/qwen3): 0.6B–32B
- [Qwen3-Coder](https://ollama.com/library/qwen3-coder): 30B
- [Qwen2.5-Coder](https://ollama.com/library/qwen2.5-coder): 0.5B–32B
- [Gemma 3](https://ollama.com/library/gemma3): 1B–27B
- [GPT-OSS](https://ollama.com/library/gpt-oss): 20B and 120B
- [DeepSeek-R1](https://ollama.com/library/deepseek-r1/tags): 1.5B–70B
- [Llama 3.2](https://ollama.com/library/llama3.2): 1B and 3B
- [Llama 3.3](https://ollama.com/library/llama3.3): 70B
- [Mistral Small 3.2](https://ollama.com/library/mistral-small3.2): 24B
- [Mistral NeMo](https://ollama.com/library/mistral-nemo): 12B
- [Phi-4](https://ollama.com/library/phi4): 14B
- [Command R](https://ollama.com/library/command-r): 35B
- [OLMo 2](https://ollama.com/library/olmo2): 7B and 13B
- [Granite 3.3](https://ollama.com/library/granite3.3): 2B and 8B

Published model artifact sizes are rounded and can change. They are **not**
measured GPU allocation, context/KV memory, or guaranteed RAM requirements.
The preflight uses approximate weights, a 30% execution allocation allowance,
a 24 GiB operating-system/context reserve, a 15% download allowance and a
20 GiB disk reserve. A fit decision is only permission to *attempt* the model.
Real compatibility must be determined empirically.

## Commands

```sh
make benchmark-start                         # auto-pull eligible tags and benchmark
make benchmark-start MODELS=installed        # test all locally installed tags
make benchmark-start MODELS=qwen3.5:9b       # explicit installed model
make benchmark-status
make benchmark-logs
make benchmark-stop
make benchmark-export
```

Use `MODEL_DIR=/actual/ollama/model/storage` if the Ollama service uses a
different disk than the current Linux user's `~/.ollama/models` path. The
default connection must be local for disk/RAM safeguards to apply.

You can extend the known candidate catalog using an external JSON file:

```json
[
  {"name": "my-model:7b", "size_gb": 5.0},
  {"name": "my-model:14b", "size_gb": 10.0}
]
```

Pass `CATALOG_FILE=/path/to/extra-models.json` when starting the benchmark.
Extra entries are merged with built-in tags without duplication. Missing tags
will be attempted and, if Ollama rejects the pull, recorded as failures rather
than causing the suite to stop. No models are downloaded by non-catalog
`llm-bench run` / `compare` commands.

The benchmark cannot discover every public Ollama registry model, and may skip
a model that could fit with tighter context settings or a different quantization.
It never pulls remote cloud-only tags, and does not silently claim models have
been tested on the actual EVO-X2 prior to running.
