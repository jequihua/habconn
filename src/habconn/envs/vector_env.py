"""Gymnasium environment for vector-action habitat restoration.

VectorHabitatEnv is the canonical v1 environment. It sits on top of:
- problems/ for static problem definitions
- state/ for dynamic episode state and action mapping
- evaluators/ for backend-abstracted connectivity evaluation
- features/ for observation packing

The action space is Discrete(K) where K is the fixed candidate-set size.
Each action index maps to a candidate planning-unit slot. Invalid/padded
slots cause the episode to terminate with zero reward (fail-fast for
development). The observation is a Dict of numpy arrays (see features/packing.py).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from habconn.evaluators.base import GraphabBackend
from habconn.evaluators.reward import compute_delta_pc_reward
from habconn.features.packing import pack_observation
from habconn.problems.candidate_generation import (
    CandidateGenerator,
    CandidateRankingStrategy,
    CandidateSet,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.action_mapping import CandidateActionMapper
from habconn.state.landscape_state import LandscapeState
from habconn.state.masks import build_action_mask


class VectorHabitatEnv(gym.Env):
    """Minimal vector-action habitat restoration environment.

    Parameters
    ----------
    problem : VectorConnectivityProblem
        Static problem definition (loaded once).
    backend : GraphabBackend
        Backend for PC evaluation (CLI, Java service, or router).
    k : int
        Fixed candidate-set size (action space = Discrete(k)).
    budget : float or None
        Total restoration budget. None uses problem.default_budget.
    n_max : int or None
        Maximum planning units for observation padding.
        None uses problem.n_planning_units.
    ranking_strategy : CandidateRankingStrategy
        How candidates are ranked each step.
    random_seed : int or None
        Seed for candidate generation (if strategy is RANDOM).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        problem: VectorConnectivityProblem,
        backend: GraphabBackend,
        k: int = 10,
        budget: Optional[float] = None,
        n_max: Optional[int] = None,
        ranking_strategy: CandidateRankingStrategy = CandidateRankingStrategy.BY_PU_ID,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._problem = problem
        self._backend = backend
        self._budget = budget
        self._k = k
        self._n_max = n_max if n_max is not None else problem.n_planning_units
        self._ranking_strategy = ranking_strategy

        self._candidate_gen = CandidateGenerator(
            k=k,
            ranking_strategy=ranking_strategy,
            random_seed=random_seed,
        )
        self._action_mapper = CandidateActionMapper()

        # Gymnasium spaces — v2 observation (see features/packing.py)
        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Dict({
            # Action-level (K,)
            "action_mask": spaces.Box(0, 1, shape=(k,), dtype=np.bool_),
            "candidate_ids": spaces.Box(-1, np.iinfo(np.int32).max, shape=(k,), dtype=np.int32),
            "candidate_costs": spaces.Box(0, np.finfo(np.float32).max, shape=(k,), dtype=np.float32),
            "candidate_areas": spaces.Box(0, np.finfo(np.float32).max, shape=(k,), dtype=np.float32),
            # Node-level (N_max,)
            "selected_mask": spaces.Box(0, 1, shape=(self._n_max,), dtype=np.bool_),
            "node_mask": spaces.Box(0, 1, shape=(self._n_max,), dtype=np.bool_),
            "node_costs": spaces.Box(0, np.finfo(np.float32).max, shape=(self._n_max,), dtype=np.float32),
            "node_areas": spaces.Box(0, np.finfo(np.float32).max, shape=(self._n_max,), dtype=np.float32),
            "eligibility_mask": spaces.Box(0, 1, shape=(self._n_max,), dtype=np.bool_),
            # Global (1,)
            "remaining_budget": spaces.Box(0, np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
            "budget_fraction": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "step_count": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            "selected_fraction": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "current_pc": spaces.Box(0, np.finfo(np.float32).max, shape=(1,), dtype=np.float32),
        })

        # Episode state (set in reset)
        self._state: Optional[LandscapeState] = None
        self._candidates: Optional[object] = None
        self._current_pc: float = 0.0
        self._initial_budget: float = 1.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to the initial state.

        Evaluates the baseline landscape (no patches selected) to
        establish the initial PC value for reward computation.
        """
        super().reset(seed=seed)

        self._state = LandscapeState.initialize(
            self._problem, budget=self._budget,
        )
        # Record initial budget for budget_fraction feature
        self._initial_budget = float(self._state.remaining_budget)

        # Evaluate baseline PC
        baseline_result = self._backend.evaluate(
            self._problem, self._state, run_label="env_baseline",
        )
        self._current_pc = baseline_result.pc_value
        self._state = self._state.with_pc_value(self._current_pc)

        # Generate first candidate set
        self._candidates = self._candidate_gen.generate(
            self._problem, self._state,
        )

        obs = self._current_obs()

        info = {
            "pc_value": self._current_pc,
            "selected_pu_ids": list(self._state.selected_pu_ids),
            "step_count": 0,
            "remaining_budget": self._state.remaining_budget,
            "n_feasible": self._candidates.n_valid,
            "backend_type": baseline_result.backend_type.value,
        }

        return obs, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one restoration action.

        Parameters
        ----------
        action : int
            Index into the current candidate set [0, K).

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        if self._state is None or self._candidates is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Map action slot to planning unit
        mapping = self._action_mapper.map_action(action, self._candidates)

        if not mapping.is_valid:
            # Fail fast on invalid/padded slot during development.
            # Mark internal state as done so subsequent step() raises.
            self._state = LandscapeState(
                problem_name=self._state.problem_name,
                remaining_budget=self._state.remaining_budget,
                selected_pu_ids=list(self._state.selected_pu_ids),
                step_count=self._state.step_count,
                cached_pc_value=self._state.cached_pc_value,
                done=True,
                info={**self._state.info, "terminated_by": "invalid_action"},
            )
            self._candidates = self._make_terminal_candidates()

            info = {
                "error": f"Invalid action slot {action} (pu_id={mapping.pu_id}, "
                         f"is_valid=False)",
                "pc_value": self._current_pc,
                "selected_pu_ids": list(self._state.selected_pu_ids),
                "step_count": self._state.step_count,
                "remaining_budget": self._state.remaining_budget,
                "n_feasible": 0,
            }
            return (
                self._current_obs(),
                0.0,  # zero reward for invalid action
                True,  # terminated
                False,
                info,
            )

        pu_id = mapping.pu_id
        pc_before = self._current_pc

        # Apply action through state layer
        self._state = self._state.apply_action(self._problem, pu_id)

        # Evaluate new state through backend
        eval_result = self._backend.evaluate(
            self._problem,
            self._state,
            run_label=f"env_step_{self._state.step_count}",
        )
        self._current_pc = eval_result.pc_value
        self._state = self._state.with_pc_value(self._current_pc)

        # Compute reward
        reward_result = compute_delta_pc_reward(pc_before, self._current_pc)
        reward = reward_result.reward

        # Check termination
        terminated = self._state.done  # set by apply_action when no feasible actions remain

        # Generate new candidates for next step, or terminal empty set
        if not terminated:
            self._candidates = self._candidate_gen.generate(
                self._problem, self._state,
            )
        else:
            self._candidates = self._make_terminal_candidates()

        obs = self._current_obs()

        info = {
            "pc_value": self._current_pc,
            "pc_before": pc_before,
            "delta_pc": reward_result.delta_pc,
            "selected_pu_ids": list(self._state.selected_pu_ids),
            "last_pu_id": pu_id,
            "step_count": self._state.step_count,
            "remaining_budget": self._state.remaining_budget,
            "backend_type": eval_result.backend_type.value,
            "action_type": eval_result.action_type.value,
            "n_feasible": self._candidates.n_valid if not terminated else 0,
        }

        return obs, reward, terminated, False, info

    def _current_obs(self) -> dict[str, np.ndarray]:
        """Build observation from current state."""
        return pack_observation(
            self._problem,
            self._state,
            self._candidates,
            n_max=self._n_max,
            current_pc=self._current_pc,
            initial_budget=self._initial_budget,
        )

    @property
    def problem(self) -> VectorConnectivityProblem:
        return self._problem

    @property
    def state(self) -> Optional[LandscapeState]:
        return self._state

    @property
    def current_pc(self) -> float:
        return self._current_pc

    def action_masks(self) -> np.ndarray:
        """Return the current action mask for sb3-contrib MaskablePPO.

        Returns a boolean array of shape (K,) where True means the
        action slot is valid. Called by MaskablePPO before each action.
        """
        if self._candidates is None:
            return np.zeros(self._k, dtype=bool)
        from habconn.state.masks import build_action_mask
        return build_action_mask(self._candidates)

    def _make_terminal_candidates(self) -> CandidateSet:
        """Create an all-padded candidate set for terminal states.

        All slots are padding, action_mask is all False, so the
        terminal observation correctly reflects zero feasible actions.
        """
        return CandidateSet(
            candidate_pu_ids=[-1] * self._k,
            valid_mask=np.zeros(self._k, dtype=bool),
            scores=np.full(self._k, np.nan),
            pad_value=-1,
            strategy="terminal",
        )
