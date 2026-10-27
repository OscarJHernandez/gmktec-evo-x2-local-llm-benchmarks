from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
import pytest
from factories import fixture_experiment
from jupyter_client import KernelManager
from nbclient import NotebookClient

from llm_bench.analytical import export_snapshot
from llm_bench.persistence import load_experiments


@pytest.mark.parametrize("scenario", ["full", "unavailable", "historical"])
def test_notebook_executes_all_cells(tmp_path: Path, scenario: str) -> None:
    repo = Path(__file__).parents[1]
    if scenario == "historical":
        experiments = load_experiments([repo / "examples/synthetic-example.json"])
    else:
        experiments = [fixture_experiment(scenario == "full")]
    snapshot = export_snapshot(experiments, tmp_path / "snapshot")
    notebook = nbformat.read(repo / "notebooks/benchmark_analysis.ipynb", as_version=4)
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert runs.height > 0\n"
            "assert runs['invocation_id'].n_unique() == runs.height\n"
            "assert performance['repetitions'].sum() == runs.height\n"
            "assert gpu['invocation_id'].n_unique() == gpu.height\n"
        )
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=90, resources={"metadata": {"path": str(repo)}}
    )
    try:
        executed = client.execute(
            env={
                **os.environ,
                "BENCHMARK_EXPORT_DIR": str(snapshot),
                "BENCHMARK_INCLUDE_SYNTHETIC": "1",
            }
        )
        assert all(
            output.output_type != "error"
            for cell in executed.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        )
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def test_notebook_missing_snapshot_actionable_error(tmp_path: Path) -> None:
    repo = Path(__file__).parents[1]
    notebook = nbformat.read(repo / "notebooks/benchmark_analysis.ipynb", as_version=4)
    setup = next(c.source for c in notebook.cells if c.cell_type == "code")
    old = os.environ.get("BENCHMARK_EXPORT_DIR")
    os.environ["BENCHMARK_EXPORT_DIR"] = str(tmp_path / "deleted")
    try:
        with pytest.raises(FileNotFoundError, match="was cleaned"):
            exec(setup, {})
    finally:
        if old is None:
            os.environ.pop("BENCHMARK_EXPORT_DIR", None)
        else:
            os.environ["BENCHMARK_EXPORT_DIR"] = old
