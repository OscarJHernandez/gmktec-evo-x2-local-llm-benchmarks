# MODELS=catalog (default): download safe-fit candidates, then benchmark every model.
# MODELS=installed: benchmark everything already downloaded, without pulling.
# MODELS=qwen3.5:9b,qwen3.5:35b: benchmark only the selected installed tags.
MODELS ?= catalog
REPEAT ?= 50
WARMUP ?= 3
NUM_CTX ?= 4096
NUM_PREDICT ?= 128
THINK ?= off
RESULTS ?= results
SESSION ?= evo-x2-llm-bench
THINK_FLAG = $(if $(filter on true 1 yes,$(THINK)),--think,--no-think)
CATALOG_FILE ?=
MODEL_DIR ?=
OPTIONAL_CATALOG = $(if $(CATALOG_FILE),--catalog-file '$(CATALOG_FILE)',)
OPTIONAL_STORE = $(if $(MODEL_DIR),--model-dir '$(MODEL_DIR)',)
COMMON_ARGS = --repeat '$(REPEAT)' --warmup '$(WARMUP)' --num-ctx '$(NUM_CTX)' --num-predict '$(NUM_PREDICT)' $(THINK_FLAG) --output-dir '$(RESULTS)'
ifeq ($(MODELS),catalog)
BENCH_COMMAND = uv run llm-bench catalog-run --auto-pull $(COMMON_ARGS) $(OPTIONAL_CATALOG) $(OPTIONAL_STORE)
else ifeq ($(MODELS),installed)
BENCH_COMMAND = uv run llm-bench catalog-run --installed-only $(COMMON_ARGS) $(OPTIONAL_STORE)
else
BENCH_COMMAND = uv run llm-bench compare --models '$(MODELS)' $(COMMON_ARGS)
endif

.PHONY: setup doctor models benchmark-start benchmark-status benchmark-stop benchmark-logs benchmark-report benchmark-export check
setup:
	uv sync --extra dev
doctor:
	uv run llm-bench doctor
models:
	uv run llm-bench models
benchmark-start:
	@command -v tmux >/dev/null || { echo "tmux is required (sudo apt install tmux)"; exit 1; }
	@! tmux has-session -t "$(SESSION)" 2>/dev/null || { echo "Benchmark session already running"; exit 1; }
	@mkdir -p "$(RESULTS)"
	@tmux new-session -d -s "$(SESSION)" "PYTHONUNBUFFERED=1 $(BENCH_COMMAND) > '$(RESULTS)/benchmark.log' 2>&1"
	@echo "Started tmux session $(SESSION). Use make benchmark-status or make benchmark-logs."
benchmark-status:
	@if tmux has-session -t "$(SESSION)" 2>/dev/null; then echo "RUNNING: $(SESSION)"; else echo "STOPPED"; fi
benchmark-stop:
	@if tmux has-session -t "$(SESSION)" 2>/dev/null; then tmux kill-session -t "$(SESSION)"; echo "Stopped benchmark; Ollama unaffected"; else echo "No benchmark session running"; fi
benchmark-logs:
	@tail -n 60 "$(RESULTS)/benchmark.log"
benchmark-report:
	uv run llm-bench report --input "$(RESULTS)" --output "$(RESULTS)/report.html"
benchmark-export:
	@mkdir -p "$(RESULTS)/exports"
	uv run llm-bench report --input "$(RESULTS)" --output "$(RESULTS)/exports/report.html" --observations "$(RESULTS)/exports/observations-$$(date -u +%Y%m%dT%H%M%S%N).csv" --jsonl "$(RESULTS)/exports/observations-$$(date -u +%Y%m%dT%H%M%S%N).jsonl"
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest
