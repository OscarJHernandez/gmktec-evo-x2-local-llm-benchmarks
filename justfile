# Task runner for local-llm-benchmarks. Run `just` to list recipes.
# Override any setting before the recipe name or via the environment:
#   just REPEAT=100 benchmark-start      REPEAT=100 just benchmark-start
#
# MODELS=catalog (default): discover library variants, download eligible models, benchmark.
# MODELS=installed: benchmark everything already downloaded, without pulling.
# MODELS=qwen3.5:9b,qwen3.5:35b: benchmark only the selected installed tags.

set shell := ["bash", "-euo", "pipefail", "-c"]

MODELS := env("MODELS", "catalog")
REPEAT := env("REPEAT", "50")
WARMUP := env("WARMUP", "3")
NUM_CTX := env("NUM_CTX", "4096")
NUM_PREDICT := env("NUM_PREDICT", "128")
THINK := env("THINK", "off")
RESULTS := env("RESULTS", "results")
JUPYTER_HOST := env("JUPYTER_HOST", "0.0.0.0")
JUPYTER_PORT := env("JUPYTER_PORT", "8888")
TELEMETRY := env("TELEMETRY", "on")
TELEMETRY_INTERVAL := env("TELEMETRY_INTERVAL", "1")
DRY_RUN := env("DRY_RUN", "0")
SESSION := env("SESSION", "evo-x2-llm-bench")
DISCOVERY := env("DISCOVERY", "library")
CATALOG_FILE := env("CATALOG_FILE", "")
MODEL_DIR := env("MODEL_DIR", "")
PHYSICS_DATASET := env("PHYSICS_DATASET", "scibench")
PHYSICS_MODELS := env("PHYSICS_MODELS", "installed")
PHYSICS_REPEAT := env("PHYSICS_REPEAT", "3")
PHYSICS_QUESTIONS := env("PHYSICS_QUESTIONS", "50")
PHYSICS_CTX := env("PHYSICS_CTX", "8192")
PHYSICS_TOKENS := env("PHYSICS_TOKENS", "1024")
PHYSICS_TEMPERATURE := env("PHYSICS_TEMPERATURE", "0.3")
PHYSICS_THINK := env("PHYSICS_THINK", "off")
PHYSICS_SESSION := env("PHYSICS_SESSION", "evo-x2-physics-bench")

_on := '^(on|true|1|yes)$'
TELEMETRY_FLAG := if TELEMETRY =~ _on { "--telemetry" } else { "--no-telemetry" }
THINK_FLAG := if THINK =~ _on { "--think" } else { "--no-think" }
PHYSICS_THINK_FLAG := if PHYSICS_THINK =~ _on { "--think" } else { "--no-think" }
CLEAN_FLAG := if DRY_RUN =~ _on { "--dry-run" } else { "" }
OPTIONAL_CATALOG := if CATALOG_FILE != "" { "--catalog-file '" + CATALOG_FILE + "'" } else { "" }
OPTIONAL_STORE := if MODEL_DIR != "" { "--model-dir '" + MODEL_DIR + "'" } else { "" }

COMMON_ARGS := TELEMETRY_FLAG + " --telemetry-interval '" + TELEMETRY_INTERVAL + "' --repeat '" + REPEAT + "' --warmup '" + WARMUP + "' --num-ctx '" + NUM_CTX + "' --num-predict '" + NUM_PREDICT + "' " + THINK_FLAG + " --output-dir '" + RESULTS + "'"

BENCH_COMMAND := if MODELS == "catalog" {
    "uv run llm-bench catalog-run --auto-pull --discovery '" + DISCOVERY + "' " + COMMON_ARGS + " " + OPTIONAL_CATALOG + " " + OPTIONAL_STORE
} else if MODELS == "installed" {
    "uv run llm-bench catalog-run --installed-only " + COMMON_ARGS + " " + OPTIONAL_STORE
} else {
    "uv run llm-bench compare --models '" + MODELS + "' " + COMMON_ARGS
}

PHYSICS_COMMAND := "uv run llm-bench physics-run --dataset '" + PHYSICS_DATASET + "' --models '" + PHYSICS_MODELS + "' --repeat '" + PHYSICS_REPEAT + "' --max-questions '" + PHYSICS_QUESTIONS + "' --num-ctx '" + PHYSICS_CTX + "' --num-predict '" + PHYSICS_TOKENS + "' --temperature '" + PHYSICS_TEMPERATURE + "' " + PHYSICS_THINK_FLAG + " --fetch --data-dir '" + RESULTS + "/datasets' --output-dir '" + RESULTS + "/physics'"

# List available recipes.
default:
    @just --list

# Install runtime, dev and analysis dependencies.
setup:
    uv sync --extra analysis

# Check Python, system resources and Ollama connectivity.
doctor:
    uv run llm-bench doctor

# List locally installed models.
models:
    uv run llm-bench models

# Preview benchmark candidates and fit checks (no downloads or inference).
benchmark-plan:
    {{BENCH_COMMAND}} --dry-run

# Start a benchmark sweep in a background tmux session.
benchmark-start:
    @command -v tmux >/dev/null || { echo "tmux is required (sudo apt install tmux)"; exit 1; }
    @! tmux has-session -t "{{SESSION}}" 2>/dev/null || { echo "Benchmark session already running"; exit 1; }
    @mkdir -p "{{RESULTS}}"
    @tmux new-session -d -s "{{SESSION}}" "PYTHONUNBUFFERED=1 {{BENCH_COMMAND}} > '{{RESULTS}}/benchmark.log' 2>&1"
    @echo "Started tmux session {{SESSION}}. Use just benchmark-status or just benchmark-logs."

# Show session state and benchmark progress.
benchmark-status:
    @if tmux has-session -t "{{SESSION}}" 2>/dev/null; then echo "tmux session: RUNNING ({{SESSION}})"; else echo "tmux session: STOPPED"; fi
    @uv run llm-bench status --output-dir "{{RESULTS}}"

# Stop the benchmark session (Ollama is unaffected).
benchmark-stop:
    @if tmux has-session -t "{{SESSION}}" 2>/dev/null; then tmux kill-session -t "{{SESSION}}"; echo "Stopped benchmark; Ollama unaffected"; else echo "No benchmark session running"; fi

# Show the last lines of the benchmark log.
benchmark-logs:
    @tail -n 60 "{{RESULTS}}/benchmark.log"

# Build an offline HTML report from experiment JSON.
benchmark-report:
    uv run llm-bench report --input "{{RESULTS}}" --output "{{RESULTS}}/report.html"

# Export a Parquet/CSV/JSONL snapshot for analysis.
benchmark-export:
    uv run llm-bench export --input "{{RESULTS}}" --output-dir "{{RESULTS}}/exports"

# Remove recognised results (preview first with DRY_RUN=1).
benchmark-clean:
    @if command -v tmux >/dev/null && tmux has-session -t "{{SESSION}}" 2>/dev/null; then echo "Stop the benchmark first: just benchmark-stop"; exit 1; fi
    uv run llm-bench clean --output-dir "{{RESULTS}}" {{CLEAN_FLAG}}

# Open the analysis notebook in JupyterLab.
notebook:
    uv run --extra analysis jupyter lab notebooks/benchmark_analysis.ipynb

# Serve JupyterLab to other devices on a trusted network.
notebook-serve:
    uv run --extra analysis jupyter lab notebooks/benchmark_analysis.ipynb --no-browser \
        --ServerApp.ip='{{JUPYTER_HOST}}' --ServerApp.port='{{JUPYTER_PORT}}' \
        --ServerApp.port_retries=0 --ServerApp.allow_remote_access=True \
        --ServerApp.root_dir='{{justfile_directory()}}'

# Physics jobs have their own tmux session and do not interfere with inference benchmarks.

# Download and cache a physics dataset.
physics-prepare:
    uv run llm-bench physics-prepare --dataset '{{PHYSICS_DATASET}}' --data-dir '{{RESULTS}}/datasets'

# Start a physics evaluation in a background tmux session.
physics-start:
    @command -v tmux >/dev/null || { echo "tmux is required (sudo apt install tmux)"; exit 1; }
    @! tmux has-session -t "{{PHYSICS_SESSION}}" 2>/dev/null || { echo "Physics benchmark already running"; exit 1; }
    @mkdir -p '{{RESULTS}}/physics'
    @tmux new-session -d -s "{{PHYSICS_SESSION}}" "PYTHONUNBUFFERED=1 {{PHYSICS_COMMAND}} > '{{RESULTS}}/physics/physics.log' 2>&1"
    @echo "Physics benchmark started in {{PHYSICS_SESSION}}"

# Show whether the physics session is running.
physics-status:
    @if tmux has-session -t "{{PHYSICS_SESSION}}" 2>/dev/null; then echo "RUNNING: {{PHYSICS_SESSION}}"; else echo "STOPPED"; fi

# Stop the physics session (Ollama is unaffected).
physics-stop:
    @if tmux has-session -t "{{PHYSICS_SESSION}}" 2>/dev/null; then tmux kill-session -t "{{PHYSICS_SESSION}}"; echo "Stopped physics evaluation; Ollama unaffected"; else echo "No physics benchmark running"; fi

# Show the last lines of the physics log.
physics-logs:
    @tail -n 60 '{{RESULTS}}/physics/physics.log'

# Summarise physics results.
physics-results:
    uv run llm-bench physics-results --dataset '{{PHYSICS_DATASET}}' --output-dir '{{RESULTS}}/physics'

# Run the same checks as CI.
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy
    uv run pytest
