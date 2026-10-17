"""Opt-in physics reasoning evaluation with reproducible, inspectable raw attempts.

Numeric scoring is a local proxy, not official UGPhysics/PHYBench/SciBench scoring.
No reference answers or solutions are included in the model prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_bench.backends.ollama import OllamaBackend, OllamaError

SCORER_VERSION = "numeric-proxy-v1"
UG_SUBJECTS = (
    "AtomicPhysics",
    "ClassicalElectromagnetism",
    "ClassicalMechanics",
    "Electrodynamics",
    "GeometricalOptics",
    "QuantumMechanics",
    "Relativity",
    "SemiconductorPhysics",
    "Solid-StatePhysics",
    "StatisticalMechanics",
    "TheoreticalMechanics",
    "Thermodynamics",
    "WaveOptics",
)
SCI_SUBJECTS = (
    "atkins",
    "calculus",
    "chemmc",
    "class",
    "diff",
    "fund",
    "matter",
    "quan",
    "stat",
    "thermo",
)
DATASETS = ("scibench", "ugphysics", "phybench")


@dataclass(frozen=True)
class Question:
    benchmark: str
    question_id: str
    topic: str
    prompt: str
    reference: Any
    unit: str | None
    dataset_sha256: str
    answer_type: str | None


def dataset_sources(dataset: str) -> dict[str, str]:
    if dataset == "scibench":
        return {
            f"{s}.json": (
                "https://raw.githubusercontent.com/mandyyyyii/scibench/main/"
                f"dataset/original/{s}.json"
            )
            for s in SCI_SUBJECTS
        }
    if dataset == "ugphysics":
        return {
            f"{s}-en.jsonl": (
                f"https://huggingface.co/datasets/UGPhysics/ugphysics/resolve/main/{s}/en.jsonl"
            )
            for s in UG_SUBJECTS
        }
    if dataset == "phybench":
        return {
            "PHYBench-fullques_v1.json": (
                "https://huggingface.co/datasets/Eureka-Lab/PHYBench/"
                "resolve/main/PHYBench-fullques_v1.json"
            )
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def prepare(dataset: str, destination: Path, *, timeout: float = 60.0) -> list[Path]:
    """Explicit dataset fetch, bounded size, no hidden downloads during inference."""
    sources = dataset_sources(dataset)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in sources.items():
        path = destination / dataset / name
        if path.is_file():
            paths.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": "local-llm-benchmarks/0.1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                # A controlled upper bound on a single source file (64 MiB).
                data = response.read(64 * 1024 * 1024 + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Cannot retrieve {dataset}/{name}: {exc}") from exc
        if len(data) > 64 * 1024 * 1024:
            raise ValueError(f"Dataset file exceeds 64 MiB: {name}")
        try:
            text = data.decode("utf-8")
            if path.suffix == ".json":
                json.loads(text)
            else:
                if not text.strip():
                    raise ValueError("Empty JSONL")
                for line in text.splitlines():
                    json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Unexpected dataset format: {name}") from exc
        # Refuse overwriting a cached file; use a temporary file in same directory.
        temp = path.with_suffix(path.suffix + ".partial")
        with temp.open("xb") as handle:
            handle.write(data)
        temp.replace(path)
        paths.append(path)
    return paths


def _entries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    parsed: Any
    if path.suffix == ".jsonl":
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Some datasets wrap lists in data / examples keys.
            for key in ("data", "examples", "questions"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
    if not isinstance(parsed, list) or any(not isinstance(x, dict) for x in parsed):
        raise ValueError(f"Expected a list of question objects: {path}")
    return parsed


def _field(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value is not None and value != "":
            return value
    return None


def load_questions(dataset: str, root: Path) -> list[Question]:
    """Read official raw files, retaining dataset checksums and excluding solutions."""
    sources = dataset_sources(dataset)
    questions: list[Question] = []
    used_ids: set[str] = set()
    for filename in sources:
        path = root / dataset / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run 'make physics-prepare PHYSICS_DATASET={dataset}'"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, entry in enumerate(_entries(path)):
            if dataset == "scibench":
                prompt = _field(entry, "problem_text")
                reference = _field(entry, "answer_number")
                qid = _field(entry, "problemid")
                topic = _field(entry, "source") or path.stem
            elif dataset == "ugphysics":
                prompt = _field(entry, "problem")
                reference = _field(entry, "answers")
                qid = _field(entry, "index")
                topic = _field(entry, "topic", "subject") or path.stem
            else:
                prompt = _field(entry, "content", "question", "problem", "Question", "problem_text")
                reference = _field(
                    entry, "reference_answer", "final_answer", "answer", "Answer", "gt"
                )
                qid = _field(entry, "id", "index", "question_id")
                topic = _field(entry, "tag", "subject", "category", "topic") or "physics"
            if not isinstance(prompt, str) or not prompt.strip():
                # Never invent questions for unknown benchmark schema.
                raise ValueError(f"Unrecognized {dataset} question field in {path}:{index}")
            unit = _field(entry, "unit")
            identifier = f"{path.stem}:{qid if qid is not None else index}"
            if identifier in used_ids:
                # Some textbook IDs repeat: make identifiers unique and stable.
                identifier = f"{identifier}:{index}"
            used_ids.add(identifier)
            questions.append(
                Question(
                    benchmark=dataset,
                    question_id=identifier,
                    topic=str(topic),
                    prompt=prompt,
                    reference=reference,
                    unit=str(unit) if unit is not None else None,
                    dataset_sha256=digest,
                    answer_type=(
                        str(entry["answer_type"]) if entry.get("answer_type") is not None else None
                    ),
                )
            )
    return questions


_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_FRACTION = re.compile(r"\\frac\{([+-]?\d+)\}\{(\d+)\}\Z")
_FINAL = re.compile(r"(?im)^\s*(?:final answer|answer)\s*:\s*(.+?)\s*$")
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")


def numeric_value(value: Any) -> float | None:
    """Only parse an explicitly unambiguous scalar, never eval generated Python."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    text = str(value).strip().strip("$").strip()
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()
    match = _FRACTION.fullmatch(text)
    if match:
        denom = int(match.group(2))
        return int(match.group(1)) / denom if denom else None
    if not _NUMBER.fullmatch(text):
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def final_numeric_answer(response: str) -> float | None:
    boxed = _BOXED.findall(response)
    if boxed:
        return numeric_value(boxed[-1])
    final = _FINAL.findall(response)
    if final:
        # Keep a numeric answer with an optional trailing physical unit.
        token = final[-1].strip().split(maxsplit=1)[0].rstrip(".,;")
        return numeric_value(token)
    return numeric_value(response)


def score(question: Question, response: str, *, rtol: float = 0.01) -> dict[str, Any]:
    """Conservative local numeric scorer, explicitly not an official benchmark score."""
    reference = numeric_value(question.reference)
    prediction = final_numeric_answer(response)
    if reference is None:
        return {
            "status": "unscored_reference",
            "correct": None,
            "prediction": prediction,
            "reference_numeric": None,
        }
    if prediction is None:
        return {
            "status": "invalid_prediction",
            "correct": False,
            "prediction": None,
            "reference_numeric": reference,
        }
    correct = math.isclose(prediction, reference, rel_tol=rtol, abs_tol=1e-9)
    return {
        "status": "scored",
        "correct": correct,
        "prediction": prediction,
        "reference_numeric": reference,
    }


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    *,
    dataset: str,
    models: list[str],
    root: Path,
    output: Path,
    repeat: int = 3,
    max_questions: int = 50,
    seed: int = 42,
    temperature: float = 0.3,
    num_ctx: int = 8192,
    num_predict: int = 1024,
    think: bool = False,
    api_url: str = "http://localhost:11434",
    backend: OllamaBackend | None = None,
) -> dict[str, int]:
    """Append one durable record per attempt, resumable with configuration identity."""
    if repeat < 1 or max_questions < 1 or not models:
        raise ValueError("Specify at least one model, one problem and one repetition")
    if not (0 <= temperature <= 2) or num_ctx < 1 or num_predict < 1:
        raise ValueError("Invalid physics generation settings")
    questions = load_questions(dataset, root)
    if not questions:
        raise ValueError(f"No problems in {dataset}")
    # Use one deterministic question subset across all models.
    rng = random.Random(seed)
    rng.shuffle(questions)
    questions = questions[:max_questions]
    client = backend or OllamaBackend(api_url)
    installed = {str(x.get("name") or x.get("model")): x for x in client.models()}
    requested = (
        sorted(name for name in installed if "embed" not in name and "cloud" not in name)
        if models == ["installed"]
        else list(dict.fromkeys(models))
    )
    if not requested:
        raise ValueError("No eligible installed models")
    missing = [name for name in requested if name not in installed]
    if missing:
        raise ValueError(f"Missing installed Ollama model(s): {', '.join(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    already: set[tuple[str, str, str, int]] = set()
    config = {
        "dataset": dataset,
        "question_limit": max_questions,
        "dataset_sha256": sorted(set(q.dataset_sha256 for q in questions)),
        "repeat": repeat,
        "seed": seed,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "think": think,
        "scorer": SCORER_VERSION,
        "question_prompt_version": 1,
    }
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            previous = json.loads(line)
            if previous.get("config_id") == config_id:
                already.add(
                    (
                        str(previous.get("model")),
                        str(previous.get("model_digest")),
                        str(previous.get("question_id")),
                        int(previous.get("repetition")),
                    )
                )
    counters = {"completed": 0, "resumed": 0, "errors": 0}
    with output.open("a", encoding="utf-8") as stream:
        for model in requested:
            details = installed[model]
            digest = str(details.get("digest") or "unknown")
            for question in questions:
                for iteration in range(1, repeat + 1):
                    key = (model, digest, question.question_id, iteration)
                    if key in already:
                        counters["resumed"] += 1
                        continue
                    prompt = (
                        "Solve the following physics problem. Explain your reasoning briefly. "
                        "Conclude with a single line 'Final answer: <numeric value>' "
                        "for numerical problems, or 'Final answer: <expression>' "
                        "for symbolic problems. "
                    )
                    if question.unit:
                        prompt += (
                            "For a numerical answer, express its coefficient in these "
                            f"units or scale: {question.unit}. "
                        )
                    prompt += "\n\n" + question.prompt
                    record: dict[str, Any] = {
                        "timestamp_utc": _utc(),
                        "benchmark": dataset,
                        "dataset_sha256": question.dataset_sha256,
                        "config_id": config_id,
                        "settings": config,
                        "scorer_version": SCORER_VERSION,
                        "model": model,
                        "model_digest": digest,
                        "quantization": (details.get("details") or {}).get("quantization_level"),
                        "question_id": question.question_id,
                        "topic": question.topic,
                        "answer_type": question.answer_type,
                        "repetition": iteration,
                        "question": question.prompt,
                        "reference_answer": question.reference,
                        "unit": question.unit,
                        "prompt": prompt,
                        "response": None,
                        "score_status": "inference_error",
                        "correct": None,
                        "predicted_numeric": None,
                        "reference_numeric": None,
                        "client_duration_seconds": None,
                        "client_ttft_seconds": None,
                        "prompt_tokens": None,
                        "output_tokens": None,
                        "prompt_eval_ns": None,
                        "eval_ns": None,
                        "server_total_ns": None,
                        "error": None,
                    }
                    try:
                        result = client.generate(
                            model,
                            prompt,
                            options={
                                "num_ctx": num_ctx,
                                "num_predict": num_predict,
                                "seed": seed + iteration - 1,
                                "temperature": temperature,
                            },
                            think=think,
                        )
                        payload = result.payload
                        response = str(payload.get("response") or "")
                        scored = score(question, response)
                        record.update(
                            {
                                "response": response,
                                "score_status": scored["status"],
                                "correct": scored["correct"],
                                "predicted_numeric": scored["prediction"],
                                "reference_numeric": scored["reference_numeric"],
                                "client_duration_seconds": result.client_duration_seconds,
                                "client_ttft_seconds": result.client_ttft_seconds,
                                "prompt_tokens": payload.get("prompt_eval_count"),
                                "output_tokens": payload.get("eval_count"),
                                "prompt_eval_ns": payload.get("prompt_eval_duration"),
                                "eval_ns": payload.get("eval_duration"),
                                "server_total_ns": payload.get("total_duration"),
                            }
                        )
                        counters["completed"] += 1
                    except (OllamaError, OSError, ValueError) as exc:
                        record["error"] = str(exc)
                        counters["errors"] += 1
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    already.add(key)
    return counters


def summary(path: Path) -> dict[tuple[str, str], dict[str, int | float | None]]:
    """Keep configurations separate; report explicit scored/unscored/error denominators."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (str(row["model"]), str(row["config_id"]))
            groups.setdefault(key, []).append(row)
    result: dict[tuple[str, str], dict[str, int | float | None]] = {}
    for key, rows in groups.items():
        eligible = [r for r in rows if r["score_status"] in ("scored", "invalid_prediction")]
        correct = sum(r["correct"] is True for r in eligible)
        result[key] = {
            "attempts": len(rows),
            "scored": len(eligible),
            "correct": correct,
            "unscored": sum(r["score_status"] == "unscored_reference" for r in rows),
            "inference_errors": sum(r["score_status"] == "inference_error" for r in rows),
            "accuracy_numeric_proxy": correct / len(eligible) if eligible else None,
        }
    return result
