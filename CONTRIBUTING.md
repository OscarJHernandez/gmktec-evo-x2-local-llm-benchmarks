# Contributing

Issues and focused pull requests are welcome. For bugs, include expected and
actual behavior plus Python, OS and Ollama versions. Do not attach raw reports
without checking for private prompts or environment information.

## Local checks

Use Python 3.10 or later and uv:

```console
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Add tests using mocked backend responses; tests must not download models, contact
a live Ollama server, or require GPU hardware. Keep raw local benchmark results
out of version control. Preserve measurement provenance and report synthetic
fixtures as synthetic.
