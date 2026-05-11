"""Unit tests for the feature-inspection helpers.

These tests exercise ``training/inspection.py`` without Graphab and
without a real SB3 model. Observations are hand-rolled dicts of
small numpy arrays that mirror the v2 contract shape.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from habconn.training.inspection import (
    ACTION_TRACE_CSV_FILENAME,
    ACTION_TRACE_JSON_FILENAME,
    FEATURE_SUMMARY_FILENAME,
    FLAT_EXTRACTOR_CONSUMED_KEYS,
    N_OBSERVATION_KEYS,
    OBSERVATION_SCHEMA_FILENAME,
    OBSERVATION_VERSION,
    build_action_trace_rows,
    observation_schema,
    summarize_observation_features,
    write_inspection_artifacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v2_observation(k: int = 4, n_max: int = 5) -> dict[str, np.ndarray]:
    """A complete v2-shaped observation dict for stub tests."""
    return {
        # Action-level (K,)
        "action_mask": np.array([True, True, False, False][:k], dtype=np.bool_),
        "candidate_ids": np.array([1, 2, -1, -1][:k], dtype=np.int32),
        "candidate_costs": np.array([1.0, 2.0, 0.0, 0.0][:k], dtype=np.float32),
        "candidate_areas": np.array([10.0, 20.0, 0.0, 0.0][:k], dtype=np.float32),
        # Node-level (N_max,)
        "selected_mask": np.zeros((n_max,), dtype=np.bool_),
        "node_mask": np.ones((n_max,), dtype=np.bool_),
        "node_costs": np.ones((n_max,), dtype=np.float32),
        "node_areas": np.full((n_max,), 10.0, dtype=np.float32),
        "eligibility_mask": np.ones((n_max,), dtype=np.bool_),
        # Global (1,)
        "remaining_budget": np.array([3.0], dtype=np.float32),
        "budget_fraction": np.array([1.0], dtype=np.float32),
        "step_count": np.array([0], dtype=np.int32),
        "selected_fraction": np.array([0.0], dtype=np.float32),
        "current_pc": np.array([2e-5], dtype=np.float32),
    }


def _make_trace_step(
    *, step: int, k: int, chosen_slot: int, pc_after: float
) -> dict:
    return {
        "step": step,
        "chosen_slot": chosen_slot,
        "chosen_pu_id": chosen_slot + 10,
        "valid_action_count": k - 1,  # last slot is padded/invalid
        "remaining_budget_before": 3.0 - (step - 1),
        "current_pc_before": pc_after - 1e-5,
        "reward_after": 1e-5,
        "pc_after": pc_after,
        "selected_pu_ids_after": list(range(10, 10 + step)),
        "action_mask": np.array([True] * (k - 1) + [False], dtype=np.bool_),
        "candidate_ids": np.array(list(range(10, 10 + k - 1)) + [-1], dtype=np.int32),
        "candidate_costs": np.array(
            [1.0 + i for i in range(k - 1)] + [0.0], dtype=np.float32,
        ),
        "candidate_areas": np.array(
            [10.0 + i for i in range(k - 1)] + [0.0], dtype=np.float32,
        ),
    }


# ---------------------------------------------------------------------------
# observation_schema
# ---------------------------------------------------------------------------


class TestObservationSchema:
    def test_has_all_v2_keys(self):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        assert schema["observation_version"] == OBSERVATION_VERSION
        assert schema["n_keys"] == N_OBSERVATION_KEYS
        names = {k["name"] for k in schema["keys"]}
        assert names == set(obs.keys())

    def test_consumed_keys_match_flat_extractor(self):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        consumed_names = {
            k["name"] for k in schema["keys"] if k["consumed_by_flat_extractor"]
        }
        assert consumed_names == set(FLAT_EXTRACTOR_CONSUMED_KEYS)
        assert set(schema["flat_extractor_consumed_keys"]) == set(
            FLAT_EXTRACTOR_CONSUMED_KEYS
        )

    def test_node_level_keys_marked_unused_with_warning(self):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        node_records = [k for k in schema["keys"] if k["group"] == "node"]
        # All node-level keys are unused by the current flat extractor.
        assert node_records, "expected node-level keys in v2 observation"
        for k in node_records:
            assert k["consumed_by_flat_extractor"] is False
        assert any("node-level arrays" in w.lower() for w in schema["warnings"])

    def test_unused_list_contains_node_level_keys(self):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        unused = set(schema["unused_by_flat_extractor"])
        # candidate_ids is action-level but unused; node-level keys also unused.
        assert "candidate_ids" in unused
        for nkey in ("selected_mask", "node_mask", "node_costs",
                     "node_areas", "eligibility_mask"):
            assert nkey in unused

    def test_schema_is_json_serializable(self):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        text = json.dumps(schema)
        round = json.loads(text)
        assert round["observation_version"] == OBSERVATION_VERSION
        assert round["n_keys"] == N_OBSERVATION_KEYS


# ---------------------------------------------------------------------------
# summarize_observation_features
# ---------------------------------------------------------------------------


class TestSummarizeObservationFeatures:
    def test_bool_arrays_report_true_false_counts(self):
        obs = _make_v2_observation()
        summary = summarize_observation_features(obs)
        am = summary["action_mask"]
        assert am["true_count"] + am["false_count"] == am["shape"][0]
        assert am["true_count"] == int(obs["action_mask"].sum())
        assert "min" not in am  # numeric-only fields not reported for bool

    def test_numeric_arrays_report_min_max_mean(self):
        obs = _make_v2_observation()
        summary = summarize_observation_features(obs)
        costs = summary["candidate_costs"]
        assert costs["min"] == pytest.approx(0.0)
        assert costs["max"] == pytest.approx(2.0)
        assert costs["mean"] == pytest.approx(np.mean(obs["candidate_costs"]))
        assert costs["finite_count"] == obs["candidate_costs"].size
        assert costs["nan_count"] == 0

    def test_nan_count_for_floats(self):
        obs = _make_v2_observation()
        # Inject a NaN into a float array.
        obs["candidate_costs"] = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
        summary = summarize_observation_features(obs)
        costs = summary["candidate_costs"]
        assert costs["nan_count"] == 1
        assert costs["finite_count"] == 3
        # Min/max/mean computed over the finite values only.
        assert costs["min"] == pytest.approx(1.0)
        assert costs["max"] == pytest.approx(4.0)

    def test_summary_is_json_serializable(self):
        obs = _make_v2_observation()
        summary = summarize_observation_features(obs)
        json.dumps(summary)  # should not raise


# ---------------------------------------------------------------------------
# build_action_trace_rows
# ---------------------------------------------------------------------------


class TestBuildActionTraceRows:
    def test_one_row_per_slot_per_step(self):
        k = 4
        steps = [
            _make_trace_step(step=1, k=k, chosen_slot=0, pc_after=3e-5),
            _make_trace_step(step=2, k=k, chosen_slot=1, pc_after=4e-5),
        ]
        rows = build_action_trace_rows(steps)
        assert len(rows) == len(steps) * k

    def test_step_and_slot_order_preserved(self):
        k = 3
        steps = [
            _make_trace_step(step=1, k=k, chosen_slot=0, pc_after=3e-5),
            _make_trace_step(step=2, k=k, chosen_slot=1, pc_after=4e-5),
        ]
        rows = build_action_trace_rows(steps)
        pairs = [(r["step"], r["slot"]) for r in rows]
        assert pairs == [
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]

    def test_chosen_marked_exactly_once_per_step(self):
        k = 4
        steps = [
            _make_trace_step(step=1, k=k, chosen_slot=2, pc_after=3e-5),
            _make_trace_step(step=2, k=k, chosen_slot=0, pc_after=4e-5),
        ]
        rows = build_action_trace_rows(steps)
        for step in (1, 2):
            chosen_for_step = [r for r in rows if r["step"] == step and r["chosen"]]
            assert len(chosen_for_step) == 1

    def test_invalid_padded_slot_marked(self):
        k = 4
        # The trace-step helper marks slot k-1 as invalid (False mask, pu_id -1).
        steps = [_make_trace_step(step=1, k=k, chosen_slot=0, pc_after=3e-5)]
        rows = build_action_trace_rows(steps)
        last = next(r for r in rows if r["slot"] == k - 1)
        assert last["valid"] is False
        assert last["pu_id"] == -1


# ---------------------------------------------------------------------------
# write_inspection_artifacts
# ---------------------------------------------------------------------------


class TestWriteInspectionArtifacts:
    def test_writes_all_four_artifacts(self, tmp_path):
        obs = _make_v2_observation()
        schema = observation_schema(obs)
        feat = {
            "run_name": "unit",
            "data_dir": "/x",
            "budget": 3,
            "k": 4,
            "n_planning_units": 5,
            "observation_version": OBSERVATION_VERSION,
            "initial_observation": summarize_observation_features(obs),
        }
        steps = [
            _make_trace_step(step=1, k=4, chosen_slot=0, pc_after=3e-5),
            _make_trace_step(step=2, k=4, chosen_slot=1, pc_after=4e-5),
        ]
        result = write_inspection_artifacts(
            tmp_path, schema=schema, feature_summary=feat,
            trace_steps=steps, run_name="unit",
        )
        for name in (
            OBSERVATION_SCHEMA_FILENAME,
            FEATURE_SUMMARY_FILENAME,
            ACTION_TRACE_JSON_FILENAME,
            ACTION_TRACE_CSV_FILENAME,
        ):
            assert (tmp_path / name).exists()

        assert result["n_deployment_trace_steps"] == 2
        assert result["n_deployment_trace_rows"] == 2 * 4

        # CSV has one row per (step, slot) plus a header.
        with (tmp_path / ACTION_TRACE_CSV_FILENAME).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 * 4

        # JSON round-trips and reflects the same counts.
        payload = json.loads((tmp_path / ACTION_TRACE_JSON_FILENAME).read_text(encoding="utf-8"))
        assert payload["n_steps"] == 2
        assert payload["n_rows"] == 2 * 4
        assert payload["run_name"] == "unit"

    def test_empty_trace_writes_valid_artifacts(self, tmp_path):
        obs = _make_v2_observation()
        result = write_inspection_artifacts(
            tmp_path,
            schema=observation_schema(obs),
            feature_summary={"observation_version": OBSERVATION_VERSION,
                              "initial_observation": summarize_observation_features(obs)},
            trace_steps=[],
        )
        assert result["n_deployment_trace_steps"] == 0
        assert result["n_deployment_trace_rows"] == 0
        # CSV header still present.
        text = (tmp_path / ACTION_TRACE_CSV_FILENAME).read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        for col in ("step", "slot", "chosen", "valid",
                     "candidate_cost", "candidate_area"):
            assert col in first_line
        # JSON has an empty steps list.
        payload = json.loads((tmp_path / ACTION_TRACE_JSON_FILENAME).read_text(encoding="utf-8"))
        assert payload["steps"] == []


# ---------------------------------------------------------------------------
# ExperimentPaths regression for the new layout
# ---------------------------------------------------------------------------


class TestExperimentPathsHasInspectionFields:
    def test_layout(self, tmp_path):
        from habconn.training.experiment import ExperimentConfig, ExperimentPaths
        cfg = ExperimentConfig(
            run_name="ok",
            data_dir=tmp_path / "d",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "out",
        )
        paths = ExperimentPaths.from_config(cfg)
        run_dir = tmp_path / "out" / "ok"
        assert paths.inspection_dir == run_dir / "inspection"
        assert paths.observation_schema_path == run_dir / "inspection" / OBSERVATION_SCHEMA_FILENAME
        assert paths.feature_summary_path == run_dir / "inspection" / FEATURE_SUMMARY_FILENAME
        assert paths.deployment_action_trace_json_path == run_dir / "inspection" / ACTION_TRACE_JSON_FILENAME
        assert paths.deployment_action_trace_csv_path == run_dir / "inspection" / ACTION_TRACE_CSV_FILENAME

    def test_ensure_creates_inspection_dir(self, tmp_path):
        from habconn.training.experiment import ExperimentConfig, ExperimentPaths
        cfg = ExperimentConfig(
            run_name="ok",
            data_dir=tmp_path / "d",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "out",
        )
        paths = ExperimentPaths.from_config(cfg)
        paths.ensure()
        assert paths.inspection_dir.is_dir()
