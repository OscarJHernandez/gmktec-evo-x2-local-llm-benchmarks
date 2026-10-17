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
PHYSICS_DATASET ?= scibench
PHYSICS_MODELS ?= installed
PHYSICS_REPEAT ?= 3
PHYSICS_QUESTIONS ?= 50
PHYSICS_CTX ?= 8192
PHYSICS_TOKENS ?= 1024
PHYSICS_TEMPERATURE ?= 0.3
PHYSICS_THINK ?= off
PHYSICS_SESSION ?= evo-x2-physics-bench
PHYSICS_THINK_FLAG = $(if $(filter on true 1 yes,$(PHYSICS_THINK)),--think,--no-think)
PHYSICS_COMMAND = uv run llm-bench physics-run --dataset '$(PHYSICS_DATASET)' --models '$(PHYSICS_MODELS)' --repeat '$(PHYSICS_REPEAT)' --max-questions '$(PHYSICS_QUESTIONS)' --num-ctx '$(PHYSICS_CTX)' --num-predict '$(PHYSICS_TOKENS)' --temperature '$(PHYSICS_TEMPERATURE)' $(PHYSICS_THINK_FLAG) --fetch --data-dir '$(RESULTS)/datasets' --output-dir '$(RESULTS)/physics'
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

.PHONY: setup doctor models benchmark-start benchmark-status benchmark-stop benchmark-logs benchmark-report benchmark-export physics-prepare physics-start physics-status physics-stop physics-logs physics-results check
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
# Physics jobs have their own tmux session and do not interfere with inference benchmarks.
physics-prepare:
	uv run llm-bench physics-prepare --dataset '$(PHYSICS_DATASET)' --data-dir '$(RESULTS)/datasets'
physics-start:
	@command -v tmux >/dev/null || { echo "tmux is required (sudo apt install tmux)"; exit 1; }
	@! tmux has-session -t "$(PHYSICS_SESSION)" 2>/dev/null || { echo "Physics benchmark already running"; exit 1; }
	@mkdir -p '$(RESULTS)/physics'
	@tmux new-session -d -s "$(PHYSICS_SESSION)" "PYTHONUNBUFFERED=1 $(PHYSICS_COMMAND) > '$(RESULTS)/physics/physics.log' 2>&1"
	@echo "Physics benchmark started in $(PHYSICS_SESSION)"
physics-status:
	@if tmux has-session -t "$(PHYSICS_SESSION)" 2>/dev/null; then echo "RUNNING: $(PHYSICS_SESSION)"; else echo "STOPPED"; fi
physics-stop:
	@if tmux has-session -t "$(PHYSICS_SESSION)" 2>/dev/null; then tmux kill-session -t "$(PHYSICS_SESSION)"; echo "Stopped physics evaluation; Ollama unaffected"; else echo "No physics benchmark running"; fi
physics-logs:
	@tail -n 60 '$(RESULTS)/physics/physics.log'
physics-results:
	uv run llm-bench physics-results --dataset '$(PHYSICS_DATASET)' --output-dir '$(RESULTS)/physics'

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest
