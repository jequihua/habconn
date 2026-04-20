"""Unit tests for the v2 observation packing and feature builders.

Uses the real bundled small_vector_001 problem for realistic fixtures
(no Graphab calls — just problem loading). Tests the deterministic
shape/key contract of pack_observation() and each feature builder.
"""

from pathlib import Path

import numpy as np
import pytest

from habconn.features.candidate_features import (
    build_candidate_areas,
    build_candidate_costs,
    build_candidate_scores,
)
from habconn.features.global_features import build_global_features
from habconn.features.node_features import (
    build_eligibility_mask,
    build_node_areas,
    build_node_costs,
)
from habconn.features.packing import pack_observation
from habconn.problems.candidate_generation import CandidateGenerator, CandidateSet
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "examples" / "small_vector_001"


def _skip_if_missing():
    if not (_data_dir() / "candidates.shp").exists():
        pytest.skip("Example data not found")


def _make_problem() -> VectorConnectivityProblem:
    d = _data_dir()
    return VectorConnectivityProblem.from_files(
        name="small_vector_001",
        vector_path=d / "candidates.shp",
        habitat_raster_path=d / "habitat.tif",
        resistance_raster_path=d / "resistance.tif",
        id_column="lyr_1",
        area_column="area",
        uniform_cost=1.0,
    )


# =============================================================================
# Node feature builders
# =============================================================================


class TestNodeFeatures:
    def test_node_costs_shape_and_pad(self):
        _skip_if_missing()
        problem = _make_problem()
        n_max = problem.n_planning_units + 5
        costs = build_node_costs(problem, n_max=n_max)

        assert costs.shape == (n_max,)
        assert costs.dtype == np.float32
        # Padded slots are 0
        assert (costs[problem.n_planning_units:] == 0).all()
        # Real slots use the cost column (uniform_cost=1.0 in fixture)
        assert (costs[: problem.n_planning_units] == 1.0).all()

    def test_node_areas_shape_and_values(self):
        _skip_if_missing()
        problem = _make_problem()
        n_max = problem.n_planning_units
        areas = build_node_areas(problem, n_max=n_max)

        assert areas.shape == (n_max,)
        assert areas.dtype == np.float32
        # All areas must be positive
        assert (areas > 0).all()

    def test_eligibility_mask(self):
        _skip_if_missing()
        problem = _make_problem()
        n_max = problem.n_planning_units + 3
        elig = build_eligibility_mask(problem, n_max=n_max)

        assert elig.shape == (n_max,)
        assert elig.dtype == bool
        # v1 fixture: all eligible
        assert elig[: problem.n_planning_units].all()
        # Padded slots are False
        assert not elig[problem.n_planning_units:].any()

    def test_n_max_too_small_raises(self):
        _skip_if_missing()
        problem = _make_problem()
        with pytest.raises(ValueError):
            build_node_costs(problem, n_max=problem.n_planning_units - 1)


# =============================================================================
# Candidate feature builders
# =============================================================================


class TestCandidateFeatures:
    def _candidate_set(self, k=5, n_valid=3, pu_ids=None):
        pu_ids = pu_ids or [1, 2, 3]
        ids = pu_ids + [-1] * (k - n_valid)
        mask = np.array([True] * n_valid + [False] * (k - n_valid), dtype=bool)
        scores = np.array([0.5, 0.4, 0.3] + [np.nan] * (k - n_valid), dtype=float)
        return CandidateSet(
            candidate_pu_ids=ids,
            valid_mask=mask,
            scores=scores,
            pad_value=-1,
            strategy="by_pu_id",
        )

    def test_candidate_costs(self):
        _skip_if_missing()
        problem = _make_problem()
        cs = self._candidate_set()
        costs = build_candidate_costs(problem, cs)

        assert costs.shape == (cs.k,)
        assert costs.dtype == np.float32
        # Valid slots have real costs; padded are 0
        assert costs[0] == 1.0
        assert costs[3] == 0.0
        assert costs[4] == 0.0

    def test_candidate_areas(self):
        _skip_if_missing()
        problem = _make_problem()
        cs = self._candidate_set()
        areas = build_candidate_areas(problem, cs)

        assert areas.shape == (cs.k,)
        assert areas.dtype == np.float32
        # Valid slots have positive areas
        assert areas[0] > 0
        assert areas[1] > 0
        # Padded slots are 0
        assert areas[3] == 0.0

    def test_candidate_scores_no_nan(self):
        _skip_if_missing()
        cs = self._candidate_set()
        scores = build_candidate_scores(cs)

        assert scores.shape == (cs.k,)
        assert np.isfinite(scores).all()
        # Valid slots preserve scores; padded replaced with 0
        assert abs(scores[0] - 0.5) < 1e-6
        assert scores[3] == 0.0


# =============================================================================
# Global feature builder
# =============================================================================


class TestGlobalFeatures:
    def test_global_features_keys_and_shapes(self):
        _skip_if_missing()
        problem = _make_problem()
        state = LandscapeState.initialize(problem, budget=3)
        g = build_global_features(
            problem, state, current_pc=2e-5, initial_budget=3.0,
        )

        expected_keys = {
            "remaining_budget", "budget_fraction",
            "step_count", "selected_fraction", "current_pc",
        }
        assert set(g.keys()) == expected_keys
        for k, v in g.items():
            assert v.shape == (1,), f"{k} shape {v.shape}"

    def test_budget_fraction_bounds(self):
        _skip_if_missing()
        problem = _make_problem()
        state = LandscapeState.initialize(problem, budget=3)

        # Full budget → fraction = 1.0
        g = build_global_features(problem, state, current_pc=0, initial_budget=3.0)
        assert abs(g["budget_fraction"][0] - 1.0) < 1e-6

        # After applying an action with cost 1, budget_fraction = 2/3
        pu_id = problem.planning_unit_ids[0]
        state2 = state.apply_action(problem, pu_id)
        g2 = build_global_features(problem, state2, current_pc=0, initial_budget=3.0)
        assert abs(g2["budget_fraction"][0] - 2.0 / 3.0) < 1e-6
        # selected_fraction: 1 / n_planning_units
        assert abs(g2["selected_fraction"][0] - 1.0 / problem.n_planning_units) < 1e-6


# =============================================================================
# pack_observation integration
# =============================================================================


class TestPackObservation:
    def _candidate_set(self, k=5, n_valid=3, pu_ids=None):
        pu_ids = pu_ids or [1, 2, 3]
        ids = pu_ids + [-1] * (k - n_valid)
        mask = np.array([True] * n_valid + [False] * (k - n_valid), dtype=bool)
        scores = np.array([0.5, 0.4, 0.3] + [np.nan] * (k - n_valid), dtype=float)
        return CandidateSet(
            candidate_pu_ids=ids,
            valid_mask=mask,
            scores=scores,
            pad_value=-1,
            strategy="by_pu_id",
        )

    def test_keys_and_shapes(self):
        _skip_if_missing()
        problem = _make_problem()
        state = LandscapeState.initialize(problem, budget=3)
        cs = self._candidate_set(k=5, n_valid=3)
        n_max = problem.n_planning_units

        obs = pack_observation(
            problem, state, cs, n_max=n_max, current_pc=2e-5, initial_budget=3.0,
        )

        expected = {
            "action_mask", "candidate_ids", "candidate_costs", "candidate_areas",
            "selected_mask", "node_mask", "node_costs", "node_areas",
            "eligibility_mask",
            "remaining_budget", "budget_fraction",
            "step_count", "selected_fraction", "current_pc",
        }
        assert set(obs.keys()) == expected

        # Action-level
        assert obs["action_mask"].shape == (5,)
        assert obs["candidate_areas"].shape == (5,)
        # Node-level
        assert obs["node_costs"].shape == (n_max,)
        assert obs["node_areas"].shape == (n_max,)
        assert obs["eligibility_mask"].shape == (n_max,)
        # Global
        assert obs["budget_fraction"].shape == (1,)
        assert obs["selected_fraction"].shape == (1,)

    def test_baseline_values_consistent(self):
        _skip_if_missing()
        problem = _make_problem()
        state = LandscapeState.initialize(problem, budget=3)
        cs = self._candidate_set(k=5, n_valid=3)

        obs = pack_observation(
            problem, state, cs,
            n_max=problem.n_planning_units,
            current_pc=1e-5,
            initial_budget=3.0,
        )

        # No selections at baseline
        assert not obs["selected_mask"].any()
        assert obs["selected_fraction"][0] == 0.0
        # Full budget
        assert obs["remaining_budget"][0] == 3.0
        assert abs(obs["budget_fraction"][0] - 1.0) < 1e-6
