"""Integration tests for VectorHabitatEnv on the bundled example landscape.

These tests verify the full environment contract: reset, step, observation,
reward, termination, and action validity, using the real CLI backend on
small_vector_001.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from habconn.envs.vector_env import VectorHabitatEnv
from habconn.evaluators.cli_exact_backend import CliExactBackend
from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.evaluators.graphab_runner import (
    GraphabProjectConfig,
    GraphabRunner,
    GraphabRuntimeConfig,
)
from habconn.problems.vector_problem import VectorConnectivityProblem


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _skip_if_missing():
    root = _project_root()
    data_dir = root / "data" / "examples" / "small_vector_001"
    if not (data_dir / "candidates.shp").exists():
        pytest.skip("Example data not found")
    if not (root / "tools" / "graphab.jar").exists():
        pytest.skip("Graphab jar not found")


def _make_env(budget: int = 3, k: int = 10) -> VectorHabitatEnv:
    root = _project_root()
    data_dir = root / "data" / "examples" / "small_vector_001"

    problem = VectorConnectivityProblem.from_files(
        name="small_vector_001",
        vector_path=data_dir / "candidates.shp",
        habitat_raster_path=data_dir / "habitat.tif",
        resistance_raster_path=data_dir / "resistance.tif",
        id_column="lyr_1",
        area_column="area",
        uniform_cost=1.0,
    )

    runtime_cfg = GraphabRuntimeConfig(
        graphab_jar_path=root / "tools" / "graphab.jar",
        work_root=root / "tmp" / "env_test_runs",
        java_executable="java",
        keep_workdirs=False,
        jvm_memory="4G",
    )
    project_cfg = GraphabProjectConfig()
    runner = GraphabRunner(runtime_cfg, project_cfg)
    evaluator = GraphabEvaluator(runner, enable_cache=True)
    backend = CliExactBackend(evaluator)

    return VectorHabitatEnv(
        problem=problem,
        backend=backend,
        k=k,
        budget=budget,
    )


# =============================================================================
# Reset tests
# =============================================================================


@pytest.mark.integration
class TestEnvReset:
    def test_reset_returns_valid_observation(self):
        _skip_if_missing()
        env = _make_env(budget=2, k=5)
        obs, info = env.reset()

        # Observation keys (v2 contract — see features/packing.py)
        expected_keys = {
            # Action-level
            "action_mask", "candidate_ids", "candidate_costs", "candidate_areas",
            # Node-level
            "selected_mask", "node_mask", "node_costs", "node_areas",
            "eligibility_mask",
            # Global
            "remaining_budget", "budget_fraction",
            "step_count", "selected_fraction", "current_pc",
        }
        assert set(obs.keys()) == expected_keys

        # Shapes
        assert obs["action_mask"].shape == (5,)
        assert obs["candidate_ids"].shape == (5,)
        assert obs["remaining_budget"].shape == (1,)
        assert obs["step_count"][0] == 0
        assert obs["remaining_budget"][0] == 2.0

    def test_reset_evaluates_baseline_pc(self):
        _skip_if_missing()
        env = _make_env(budget=2)
        obs, info = env.reset()

        assert "pc_value" in info
        assert math.isfinite(info["pc_value"])
        assert info["pc_value"] > 0
        assert obs["current_pc"][0] > 0

    def test_reset_produces_valid_candidates(self):
        _skip_if_missing()
        env = _make_env(budget=2, k=5)
        obs, info = env.reset()

        # At least some candidates should be valid
        assert obs["action_mask"].any()
        assert info["n_feasible"] > 0


# =============================================================================
# Step tests
# =============================================================================


@pytest.mark.integration
class TestEnvStep:
    def test_step_updates_state(self):
        _skip_if_missing()
        env = _make_env(budget=2, k=10)
        obs0, info0 = env.reset()

        # Take first valid action
        valid_actions = np.where(obs0["action_mask"])[0]
        action = int(valid_actions[0])

        obs1, reward, terminated, truncated, info1 = env.step(action)

        assert info1["step_count"] == 1
        assert info1["remaining_budget"] == 1.0
        assert len(info1["selected_pu_ids"]) == 1
        assert info1["last_pu_id"] == obs0["candidate_ids"][action]

    def test_step_reward_is_delta_pc(self):
        _skip_if_missing()
        env = _make_env(budget=2, k=10)
        obs0, info0 = env.reset()
        pc_before = info0["pc_value"]

        valid_actions = np.where(obs0["action_mask"])[0]
        obs1, reward, terminated, truncated, info1 = env.step(int(valid_actions[0]))

        expected_delta = info1["pc_value"] - pc_before
        assert abs(reward - expected_delta) < 1e-15
        assert info1["delta_pc"] == reward

    def test_step_pc_increases_with_restoration(self):
        """Adding habitat should generally increase connectivity."""
        _skip_if_missing()
        env = _make_env(budget=2, k=10)
        obs0, info0 = env.reset()

        valid_actions = np.where(obs0["action_mask"])[0]
        obs1, reward, terminated, truncated, info1 = env.step(int(valid_actions[0]))

        assert info1["pc_value"] >= info0["pc_value"]
        assert reward >= 0

    def test_multi_step_episode(self):
        """Run a full 2-step episode and verify termination."""
        _skip_if_missing()
        env = _make_env(budget=2, k=10)
        obs, info = env.reset()

        rewards = []
        for _ in range(2):
            valid = np.where(obs["action_mask"])[0]
            if len(valid) == 0:
                break
            obs, reward, terminated, truncated, info = env.step(int(valid[0]))
            rewards.append(reward)
            if terminated:
                break

        assert len(rewards) == 2
        assert info["step_count"] == 2
        assert info["remaining_budget"] == 0.0
        # Budget exhausted → should be terminated
        assert terminated


# =============================================================================
# Invalid action tests
# =============================================================================


@pytest.mark.integration
class TestEnvInvalidAction:
    def test_invalid_padded_slot_terminates(self):
        """Choosing a padded action slot should fail fast with zero reward."""
        _skip_if_missing()
        # Use k larger than feasible candidates to ensure padding exists
        env = _make_env(budget=1, k=100)
        obs, info = env.reset()

        # Find a padded (invalid) slot
        invalid_actions = np.where(~obs["action_mask"])[0]
        if len(invalid_actions) == 0:
            pytest.skip("No padded slots available")

        obs1, reward, terminated, truncated, info1 = env.step(int(invalid_actions[0]))

        assert reward == 0.0
        assert terminated is True
        assert "error" in info1


# =============================================================================
# Termination tests
# =============================================================================


@pytest.mark.integration
class TestEnvTermination:
    def test_budget_exhaustion_terminates(self):
        _skip_if_missing()
        env = _make_env(budget=1, k=10)
        obs, info = env.reset()

        valid = np.where(obs["action_mask"])[0]
        obs, reward, terminated, truncated, info = env.step(int(valid[0]))

        # Budget was 1, cost is 1 per unit → should be done
        assert terminated is True
        assert info["remaining_budget"] == 0.0

    def test_step_after_done_raises(self):
        _skip_if_missing()
        env = _make_env(budget=1, k=10)
        obs, info = env.reset()

        valid = np.where(obs["action_mask"])[0]
        env.step(int(valid[0]))

        with pytest.raises(RuntimeError, match="done"):
            env.step(0)


# =============================================================================
# Regression tests for review-identified bugs
# =============================================================================


@pytest.mark.integration
class TestTerminalConsistency:
    """Tests for the exact bugs found in the environment review."""

    def test_invalid_action_termination_is_sticky(self):
        """After an invalid padded action returns terminated=True,
        a second step() must raise — the episode must be truly done."""
        _skip_if_missing()
        env = _make_env(budget=2, k=100)
        obs, info = env.reset()

        # Take an invalid (padded) action
        invalid = np.where(~obs["action_mask"])[0]
        if len(invalid) == 0:
            pytest.skip("No padded slots")

        obs1, reward, terminated, truncated, info1 = env.step(int(invalid[0]))
        assert terminated is True
        assert env.state.done is True  # internal state is also terminal

        # Second step must raise
        with pytest.raises(RuntimeError, match="done"):
            env.step(0)

    def test_invalid_action_terminal_obs_has_no_valid_slots(self):
        """After an invalid action terminates the episode,
        the returned observation must show no valid action slots."""
        _skip_if_missing()
        env = _make_env(budget=2, k=100)
        obs, info = env.reset()

        invalid = np.where(~obs["action_mask"])[0]
        if len(invalid) == 0:
            pytest.skip("No padded slots")

        obs1, reward, terminated, truncated, info1 = env.step(int(invalid[0]))
        assert terminated is True
        assert obs1["action_mask"].sum() == 0
        assert info1["n_feasible"] == 0

    def test_valid_terminal_obs_has_no_valid_slots(self):
        """When a valid action exhausts the budget (terminated=True),
        the returned observation must have action_mask all-False."""
        _skip_if_missing()
        env = _make_env(budget=1, k=10)
        obs, info = env.reset()

        valid = np.where(obs["action_mask"])[0]
        obs1, reward, terminated, truncated, info1 = env.step(int(valid[0]))

        assert terminated is True
        assert obs1["action_mask"].sum() == 0
        assert info1["n_feasible"] == 0

    def test_selected_unit_not_in_next_candidates(self):
        """After selecting a planning unit, it must not appear
        in the next step's candidate set."""
        _skip_if_missing()
        env = _make_env(budget=3, k=10)
        obs0, info0 = env.reset()

        valid = np.where(obs0["action_mask"])[0]
        chosen_slot = int(valid[0])
        chosen_pu_id = obs0["candidate_ids"][chosen_slot]

        obs1, reward, terminated, truncated, info1 = env.step(chosen_slot)

        if not terminated:
            # The chosen PU should not be among valid candidates anymore
            valid_ids_after = obs1["candidate_ids"][obs1["action_mask"]]
            assert chosen_pu_id not in valid_ids_after

    def test_terminal_obs_and_info_agree(self):
        """On terminal transitions, obs and info must be consistent
        about feasible action count."""
        _skip_if_missing()
        env = _make_env(budget=1, k=10)
        obs, info = env.reset()

        valid = np.where(obs["action_mask"])[0]
        obs1, reward, terminated, truncated, info1 = env.step(int(valid[0]))

        assert terminated is True
        n_valid_obs = int(obs1["action_mask"].sum())
        n_feasible_info = info1["n_feasible"]
        assert n_valid_obs == n_feasible_info == 0
