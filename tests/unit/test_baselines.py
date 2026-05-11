"""Unit tests for the simple-baseline evaluation surface.

These tests exercise the action selectors, the per-baseline evaluation
loop (against a synthetic stub env), and the comparison artifact
writer. They do not require Graphab.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from habconn.training.baselines import (
    BASELINE_METHODS,
    evaluate_largest_area,
    evaluate_lowest_cost,
    evaluate_random_valid,
    run_evaluation_comparison,
    select_largest_area,
    select_lowest_cost,
    select_random_valid,
    write_comparison_artifacts,
)
from habconn.training.evaluation import EvalEpisodeResult, EvalSummary


# ---------------------------------------------------------------------------
# Action selectors
# ---------------------------------------------------------------------------


class TestSelectRandomValid:
    def test_only_valid_slots_chosen(self):
        mask = np.array([False, True, False, True, False], dtype=bool)
        rng = np.random.default_rng(42)
        # Many draws — every output must be in the valid set.
        valid = {1, 3}
        for _ in range(50):
            choice = select_random_valid(mask, rng=rng)
            assert choice in valid

    def test_deterministic_for_same_seed(self):
        mask = np.array([True, True, True, True, True], dtype=bool)
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        seq_a = [select_random_valid(mask, rng=rng_a) for _ in range(20)]
        seq_b = [select_random_valid(mask, rng=rng_b) for _ in range(20)]
        assert seq_a == seq_b

    def test_distinct_for_different_seeds(self):
        mask = np.array([True, True, True, True, True], dtype=bool)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        seq_a = [select_random_valid(mask, rng=rng_a) for _ in range(20)]
        seq_b = [select_random_valid(mask, rng=rng_b) for _ in range(20)]
        assert seq_a != seq_b

    def test_empty_mask_rejected(self):
        mask = np.array([False, False, False], dtype=bool)
        with pytest.raises(ValueError):
            select_random_valid(mask, rng=np.random.default_rng(0))


class TestSelectLowestCost:
    def test_picks_lowest_valid(self):
        # Slot 0 has the lowest cost but is invalid; slot 2 is the next lowest.
        mask = np.array([False, True, True, True], dtype=bool)
        costs = np.array([0.0, 5.0, 1.0, 3.0], dtype=np.float32)
        assert select_lowest_cost(mask, costs) == 2

    def test_ignores_invalid_cheaper(self):
        mask = np.array([False, False, True, True], dtype=bool)
        costs = np.array([-100.0, -50.0, 10.0, 20.0], dtype=np.float32)
        assert select_lowest_cost(mask, costs) == 2

    def test_ties_break_to_first_valid(self):
        # Slots 1 and 3 tie on lowest cost; argmin returns 1.
        mask = np.array([True, True, True, True], dtype=bool)
        costs = np.array([5.0, 1.0, 5.0, 1.0], dtype=np.float32)
        assert select_lowest_cost(mask, costs) == 1

    def test_empty_mask_rejected(self):
        mask = np.array([False, False, False], dtype=bool)
        with pytest.raises(ValueError):
            select_lowest_cost(mask, np.zeros(3, dtype=np.float32))


class TestSelectLargestArea:
    def test_picks_largest_valid(self):
        # Slot 3 has the largest area but is invalid; slot 1 is next.
        mask = np.array([True, True, True, False], dtype=bool)
        areas = np.array([2.0, 9.0, 5.0, 100.0], dtype=np.float32)
        assert select_largest_area(mask, areas) == 1

    def test_ignores_invalid_larger(self):
        mask = np.array([False, False, True, True], dtype=bool)
        areas = np.array([1000.0, 500.0, 10.0, 5.0], dtype=np.float32)
        assert select_largest_area(mask, areas) == 2

    def test_ties_break_to_first_valid(self):
        mask = np.array([True, True, True, True], dtype=bool)
        areas = np.array([1.0, 9.0, 9.0, 1.0], dtype=np.float32)
        assert select_largest_area(mask, areas) == 1

    def test_empty_mask_rejected(self):
        mask = np.array([False, False, False], dtype=bool)
        with pytest.raises(ValueError):
            select_largest_area(mask, np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# Stub env for end-to-end episode runs
# ---------------------------------------------------------------------------


class _StubEnv(gym.Env):
    """Minimal multi-step env emulating VectorHabitatEnv's API surface.

    Each episode lasts ``budget`` valid steps. Costs and areas are given
    as constructor arguments. The action mask becomes all-False after
    the configured number of steps so the baselines terminate.

    PC values increase by ``pc_increment`` per step from a fixed
    baseline so that ``final_pc`` and ``delta_pc`` are deterministic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        k: int,
        budget: int,
        costs: np.ndarray,
        areas: np.ndarray,
        baseline_pc: float = 0.0001,
        pc_increment: float = 0.00001,
    ) -> None:
        super().__init__()
        self.k = k
        self.budget_total = budget
        self.costs = costs.astype(np.float32)
        self.areas = areas.astype(np.float32)
        self.baseline_pc = baseline_pc
        self.pc_increment = pc_increment
        self.observation_space = gym.spaces.Dict({
            "candidate_costs": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
            "candidate_areas": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(k)
        self._step_idx = 0
        self._selected: list[int] = []
        self._last_pc = baseline_pc

    def _obs(self) -> dict:
        return {
            "candidate_costs": self.costs.copy(),
            "candidate_areas": self.areas.copy(),
        }

    def reset(self, *, seed=None, options=None):
        self._step_idx = 0
        self._selected = []
        self._last_pc = self.baseline_pc
        return self._obs(), {"pc_value": self.baseline_pc, "selected_pu_ids": []}

    def action_masks(self) -> np.ndarray:
        if self._step_idx >= self.budget_total:
            return np.zeros((self.k,), dtype=bool)
        return np.ones((self.k,), dtype=bool)

    def step(self, action: int):
        self._step_idx += 1
        self._selected.append(int(action))
        self._last_pc = self.baseline_pc + self.pc_increment * self._step_idx
        terminated = self._step_idx >= self.budget_total
        info = {"pc_value": self._last_pc, "selected_pu_ids": list(self._selected)}
        # Reward is delta-PC; for this stub it equals pc_increment per step.
        return self._obs(), self.pc_increment, terminated, False, info


def _make_stub_env(k: int = 4, budget: int = 3) -> _StubEnv:
    return _StubEnv(
        k=k,
        budget=budget,
        costs=np.array([5.0, 1.0, 3.0, 7.0], dtype=np.float32),
        areas=np.array([2.0, 9.0, 5.0, 1.0], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Per-baseline evaluation entry points
# ---------------------------------------------------------------------------


class TestEvaluateBaselines:
    def test_lowest_cost_picks_slot_1_each_step(self):
        env = _make_stub_env()
        s = evaluate_lowest_cost(env, n_episodes=1)
        assert s.n_episodes == 1
        ep = s.episodes[0]
        # Costs: [5, 1, 3, 7]; lowest valid slot is 1 every step.
        assert ep.selected_pu_ids == [1, 1, 1]
        assert ep.episode_steps == 3
        assert ep.final_pc > ep.baseline_pc

    def test_largest_area_picks_slot_1_each_step(self):
        env = _make_stub_env()
        s = evaluate_largest_area(env, n_episodes=1)
        ep = s.episodes[0]
        # Areas: [2, 9, 5, 1]; largest valid slot is 1 every step.
        assert ep.selected_pu_ids == [1, 1, 1]

    def test_random_valid_is_seeded(self):
        env = _make_stub_env()
        s1 = evaluate_random_valid(env, n_episodes=1, base_seed=42)
        env = _make_stub_env()  # fresh env to compare
        s2 = evaluate_random_valid(env, n_episodes=1, base_seed=42)
        assert s1.episodes[0].selected_pu_ids == s2.episodes[0].selected_pu_ids

    def test_random_valid_distinct_for_different_seeds(self):
        env = _make_stub_env()
        s1 = evaluate_random_valid(env, n_episodes=1, base_seed=1)
        env = _make_stub_env()
        s2 = evaluate_random_valid(env, n_episodes=1, base_seed=999)
        # Not strictly guaranteed, but with 3 steps and 4 valid slots
        # collision probability is low enough that this asserts behavior
        # in practice for these specific seeds.
        assert s1.episodes[0].selected_pu_ids != s2.episodes[0].selected_pu_ids

    def test_invalid_n_episodes_rejected(self):
        env = _make_stub_env()
        with pytest.raises(ValueError):
            evaluate_lowest_cost(env, n_episodes=0)
        with pytest.raises(ValueError):
            evaluate_largest_area(env, n_episodes=-1)
        with pytest.raises(ValueError):
            evaluate_random_valid(env, n_episodes=0, base_seed=1)


# ---------------------------------------------------------------------------
# Comparison writer
# ---------------------------------------------------------------------------


def _make_summary(name: str) -> EvalSummary:
    ep = EvalEpisodeResult(
        episode_return=1e-6,
        episode_steps=3,
        final_pc=2.5e-5,
        baseline_pc=2.0e-5,
        delta_pc_total=5e-6,
        selected_pu_ids=[1, 2, 3],
        step_rewards=[1e-6, 2e-6, 2e-6],
        step_pc_values=[2.1e-5, 2.3e-5, 2.5e-5],
    )
    return EvalSummary(
        n_episodes=1,
        mean_return=1e-6,
        mean_steps=3.0,
        mean_final_pc=2.5e-5,
        mean_delta_pc=5e-6,
        episodes=[ep],
    )


class TestWriteComparisonArtifacts:
    def test_writes_json_and_csv(self, tmp_path):
        summaries = {
            "trained_policy": _make_summary("trained_policy"),
            "random_valid": _make_summary("random_valid"),
            "lowest_cost": _make_summary("lowest_cost"),
            "largest_area": _make_summary("largest_area"),
        }
        write_comparison_artifacts(
            tmp_path,
            summaries,
            run_name="unit_test",
            seed=42,
            budget=3,
            k=4,
            n_eval_episodes=1,
        )
        json_path = tmp_path / "comparison.json"
        csv_path = tmp_path / "comparison.csv"
        assert json_path.exists()
        assert csv_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["run_name"] == "unit_test"
        assert payload["seed"] == 42
        assert payload["budget"] == 3
        assert payload["k"] == 4
        assert payload["n_eval_episodes"] == 1
        assert set(payload["methods"].keys()) == {
            "trained_policy",
            "random_valid",
            "lowest_cost",
            "largest_area",
        }
        # Per-method means have the expected schema.
        for name in ("trained_policy", "random_valid", "lowest_cost", "largest_area"):
            m = payload["method_means"][name]
            assert {
                "method", "n_episodes", "mean_return",
                "mean_final_pc", "mean_delta_pc", "mean_steps",
            }.issubset(m.keys())

    def test_csv_contains_all_methods(self, tmp_path):
        summaries = {
            "trained_policy": _make_summary("trained_policy"),
            "random_valid": _make_summary("random_valid"),
            "lowest_cost": _make_summary("lowest_cost"),
            "largest_area": _make_summary("largest_area"),
        }
        write_comparison_artifacts(
            tmp_path,
            summaries,
            run_name="unit_test",
            seed=42,
            budget=3,
            k=4,
            n_eval_episodes=1,
        )
        with (tmp_path / "comparison.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        names = {r["method"] for r in rows}
        assert names == {"trained_policy", "random_valid", "lowest_cost", "largest_area"}
        # Numeric columns parse as floats.
        for r in rows:
            float(r["mean_return"])
            float(r["mean_final_pc"])
            float(r["mean_delta_pc"])
            float(r["mean_steps"])


class TestRunEvaluationComparison:
    def test_against_stub_env(self, tmp_path):
        env = _make_stub_env()
        trained = _make_summary("trained_policy")  # synthesized trained eval
        comparison = run_evaluation_comparison(
            env=env,
            n_eval_episodes=1,
            base_seed=7,
            output_dir=tmp_path,
            trained_policy_summary=trained,
            run_name="unit_run",
            budget=3,
            k=4,
        )
        assert (tmp_path / "comparison.json").exists()
        assert (tmp_path / "comparison.csv").exists()
        assert set(comparison["methods"].keys()) == {
            "trained_policy",
            "random_valid",
            "lowest_cost",
            "largest_area",
        }
        # The trained-policy summary is reused exactly (no re-evaluation).
        assert comparison["methods"]["trained_policy"]["mean_final_pc"] == trained.mean_final_pc

    def test_baseline_methods_constant(self):
        assert BASELINE_METHODS == ("random_valid", "lowest_cost", "largest_area")
