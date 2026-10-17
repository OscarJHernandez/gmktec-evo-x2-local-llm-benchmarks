"""Physics suite tests use tiny synthetic fixtures and a fake Ollama backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_bench import cli
from llm_bench.backends.base import GenerateResult
from llm_bench.physics import (
    Question,
    dataset_sources,
    final_numeric_answer,
    load_questions,
    numeric_value,
    run,
    score,
    summary,
)


def _question(reference: Any = "4.5") -> Question:
    return Question(
        "scibench", "fund:1", "mechanics", "Find the force.", reference, "N", "abc", None
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Final answer: 4.5 N", 4.5),
        ("Step one\nFinal answer: 4.5", 4.5),
        (r"\boxed{4.5}", 4.5),
        ("4.5", 4.5),
        (r"Final answer: \frac{9}{2}", 4.5),
        ("A rough guess is 4.5", None),
        ("Final answer: sqrt(20)", None),
        ("Final answer: 5e1000", None),
    ],
)
def test_final_numeric_extraction(response: str, expected: float | None) -> None:
    assert final_numeric_answer(response) == expected


def test_numeric_scoring_and_reference_unscored() -> None:
    assert score(_question(), "Final answer: 4.53")["correct"] is True
    assert score(_question(), "Final answer: 6")["correct"] is False
    assert score(_question(), "no parseable answer")["status"] == "invalid_prediction"
    symbolic = _question(r"\frac{mv^2}{2}")
    result = score(symbolic, "Final answer: mv^2/2")
    assert result["status"] == "unscored_reference"
    assert result["correct"] is None
    assert numeric_value(["5"]) == 5
    assert numeric_value(["5", "6"]) is None


def test_dataset_adapters_use_correct_public_fields(tmp_path: Path) -> None:
    fixtures = {
        "scibench": (
            "fund.json",
            [{"problem_text": "Force?", "answer_number": "5", "source": "fund", "problemid": "1"}],
        ),
        "ugphysics": (
            "AtomicPhysics-en.jsonl",
            [
                {
                    "index": 44,
                    "problem": "Find the momentum",
                    "answers": ["2"],
                    "subject": "AtomicPhysics",
                    "topic": "Momentum",
                    "solution": "Do not leak the solution",
                }
            ],
        ),
        "phybench": (
            "PHYBench-fullques_v1.json",
            [
                {
                    "id": 321,
                    "content": "Derive the energy",
                    "tag": "MECHANICS",
                    "solution": "Secret derivation",
                    "answer": r"\frac12mv^2",
                }
            ],
        ),
    }
    for dataset, (filename, rows) in fixtures.items():
        # Every adapter requires all subject files; create empty placeholders.
        for file_name in dataset_sources(dataset):
            path = tmp_path / dataset / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if file_name == filename:
                if path.suffix == ".jsonl":
                    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                else:
                    path.write_text(json.dumps(rows), encoding="utf-8")
            else:
                path.write_text("[]" if path.suffix == ".json" else "", encoding="utf-8")
        questions = load_questions(dataset, tmp_path)
        assert len(questions) == 1
        assert "solution" not in questions[0].prompt.lower()
        assert questions[0].reference is not None
        assert questions[0].dataset_sha256


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def models(self) -> list[dict[str, Any]]:
        return [
            {"name": "fake:1b", "digest": "abcdef", "details": {"quantization_level": "Q4"}},
            {"name": "nomic-embed-text:latest", "digest": "embed", "details": {}},
        ]

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any],
        think: bool = False,
        keep_alive: str | int | None = None,
    ) -> GenerateResult:
        self.calls.append({"prompt": prompt, "options": options, "think": think})
        return GenerateResult(
            payload={
                "response": "Calculation follows.\nFinal answer: 4.5 N",
                "prompt_eval_count": 100,
                "eval_count": 20,
                "prompt_eval_duration": 123,
                "eval_duration": 345,
                "total_duration": 567,
            },
            client_duration_seconds=0.4,
            client_ttft_seconds=0.1,
        )


def test_attempts_persist_resume_and_separate_references(tmp_path: Path) -> None:
    base = tmp_path / "datasets"
    folder = base / "phybench"
    folder.mkdir(parents=True)
    (folder / "PHYBench-fullques_v1.json").write_text(
        json.dumps([{"id": 3, "tag": "MECHANICS", "content": "Find F.", "answer": "4.5"}]),
        encoding="utf-8",
    )
    output = tmp_path / "runs" / "attempts.jsonl"
    backend = FakeOllama()
    config: dict[str, Any] = {
        "dataset": "phybench",
        "models": ["fake:1b"],
        "root": base,
        "output": output,
        "repeat": 3,
        "max_questions": 1,
        "backend": backend,
    }
    counts = run(**config)
    assert counts == {"completed": 3, "resumed": 0, "errors": 0}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["correct"] is True for row in rows)
    assert all(row["reference_answer"] == "4.5" for row in rows)
    assert all("4.5" not in call["prompt"] for call in backend.calls)
    assert len({row["repetition"] for row in rows}) == 3
    assert len({call["options"]["seed"] for call in backend.calls}) == 3
    second = run(**config)
    assert second == {"completed": 0, "resumed": 3, "errors": 0}
    assert len(output.read_text().splitlines()) == 3
    assert list(summary(output).values())[0]["accuracy_numeric_proxy"] == 1.0


def test_phybench_symbolic_is_unscored_not_wrong(tmp_path: Path) -> None:
    root = tmp_path / "data"
    folder = root / "phybench"
    folder.mkdir(parents=True)
    (folder / "PHYBench-fullques_v1.json").write_text(
        json.dumps([{"id": 4, "content": "Find F.", "tag": "MECHANICS", "answer": "m*a"}]),
        encoding="utf-8",
    )
    output = tmp_path / "physics.jsonl"
    run(
        dataset="phybench",
        models=["fake:1b"],
        root=root,
        output=output,
        repeat=1,
        max_questions=1,
        backend=FakeOllama(),
    )
    record = json.loads(output.read_text().strip())
    assert record["score_status"] == "unscored_reference"
    assert record["correct"] is None


def test_physics_commands_opt_in_and_not_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "prepare_physics", lambda *args: [Path("fake.json")])
    result = CliRunner().invoke(cli.app, ["physics-prepare", "--dataset", "scibench"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(cli.app, ["physics-prepare", "--dataset", "not-a-benchmark"])
    assert result.exit_code != 0
