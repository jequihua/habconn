"""Notebook scaffold presence + JSON validity test.

This is a cheap test. It does **not** execute any notebook cell. It
confirms only that:

- the notebooks directory exists,
- the four expected notebook files are present,
- the README is present,
- each notebook file parses as JSON and has the minimal nbformat
  fields a Jupyter notebook is expected to expose.

Notebooks are a human use/exploration environment for the current
single-landscape workflow. They are intentionally not part of the
training, evaluation, or deployment paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PACKAGE_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = (
    "00_environment_check.ipynb",
    "01_problem_and_environment_walkthrough.ipynb",
    "02_training_smoke_run.ipynb",
    "03_outputs_evaluation_deployment_inspection.ipynb",
)


def test_notebooks_directory_exists() -> None:
    assert NOTEBOOKS_DIR.is_dir(), f"notebooks dir missing: {NOTEBOOKS_DIR}"


def test_readme_exists() -> None:
    readme = NOTEBOOKS_DIR / "README.md"
    assert readme.is_file(), f"notebooks README missing: {readme}"
    text = readme.read_text(encoding="utf-8")
    assert text.strip(), "notebooks README is empty"


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_file_exists(filename: str) -> None:
    path = NOTEBOOKS_DIR / filename
    assert path.is_file(), f"notebook missing: {path}"


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_parses_as_json(filename: str) -> None:
    path = NOTEBOOKS_DIR / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{filename}: notebook root is not an object"
    assert payload.get("nbformat") == 4, f"{filename}: nbformat != 4"
    assert isinstance(payload.get("cells"), list), f"{filename}: cells must be a list"
    assert payload["cells"], f"{filename}: notebook has no cells"
    for cell in payload["cells"]:
        assert "cell_type" in cell, f"{filename}: cell missing cell_type"
        assert cell["cell_type"] in {"markdown", "code", "raw"}, (
            f"{filename}: unexpected cell_type {cell['cell_type']!r}"
        )


def _concat_cell_sources(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_outputs_notebook_uses_flat_model_selection_keys() -> None:
    """`model_selection.json` exposes flat keys; the inspection notebook
    must read them and must not regress to the stale nested schema."""
    path = NOTEBOOKS_DIR / "03_outputs_evaluation_deployment_inspection.ipynb"
    text = _concat_cell_sources(path)
    for key in (
        "selected_candidate_id",
        "selected_candidate_type",
        "selected_candidate_timestep",
    ):
        assert key in text, f"notebook missing flat key {key!r}"
    for stale in (
        'selected_candidate"]["id',
        "selected_candidate'][\"id",
        "selected_candidate']['id",
    ):
        assert stale not in text, f"notebook still has stale access {stale!r}"


def test_outputs_notebook_iterates_schema_key_records() -> None:
    """`observation_schema.json` stores keys as a list of records."""
    path = NOTEBOOKS_DIR / "03_outputs_evaluation_deployment_inspection.ipynb"
    text = _concat_cell_sources(path)
    assert "for info in schema['keys']" in text
    assert "key = info['name']" in text
    assert "schema['keys'].items()" not in text
