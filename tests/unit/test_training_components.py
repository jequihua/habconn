"""Unit tests for training components that don't require Graphab."""

import gymnasium as gym
import numpy as np
import torch

from habconn.models.extractors.padded_mlp import FlatObsExtractor
from habconn.training.evaluation import (
    EvalEpisodeResult,
    EvalSummary,
    evaluate_policy,
)


def _make_obs_space(k=5, n_max=10) -> gym.spaces.Dict:
    """Matches VectorHabitatEnv.observation_space for the v2 contract."""
    return gym.spaces.Dict({
        # Action-level (K,)
        "action_mask": gym.spaces.Box(0, 1, shape=(k,), dtype=np.bool_),
        "candidate_ids": gym.spaces.Box(-1, 2**31 - 1, shape=(k,), dtype=np.int32),
        "candidate_costs": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
        "candidate_areas": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
        # Node-level (N_max,)
        "selected_mask": gym.spaces.Box(0, 1, shape=(n_max,), dtype=np.bool_),
        "node_mask": gym.spaces.Box(0, 1, shape=(n_max,), dtype=np.bool_),
        "node_costs": gym.spaces.Box(0, 1e6, shape=(n_max,), dtype=np.float32),
        "node_areas": gym.spaces.Box(0, 1e6, shape=(n_max,), dtype=np.float32),
        "eligibility_mask": gym.spaces.Box(0, 1, shape=(n_max,), dtype=np.bool_),
        # Global (1,)
        "remaining_budget": gym.spaces.Box(0, 1e6, shape=(1,), dtype=np.float32),
        "budget_fraction": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        "step_count": gym.spaces.Box(0, 2**31 - 1, shape=(1,), dtype=np.int32),
        "selected_fraction": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        "current_pc": gym.spaces.Box(0, 1e6, shape=(1,), dtype=np.float32),
    })


def _make_batch_obs(k=5, n_max=10, batch=2) -> dict:
    return {
        # Action-level
        "action_mask": torch.tensor(
            [[1, 1, 0, 0, 0]] * batch, dtype=torch.bool,
        )[:, :k],
        "candidate_ids": torch.zeros(batch, k, dtype=torch.int32),
        "candidate_costs": torch.ones(batch, k),
        "candidate_areas": torch.full((batch, k), 10000.0),
        # Node-level
        "selected_mask": torch.zeros(batch, n_max, dtype=torch.bool),
        "node_mask": torch.ones(batch, n_max, dtype=torch.bool),
        "node_costs": torch.ones(batch, n_max),
        "node_areas": torch.full((batch, n_max), 10000.0),
        "eligibility_mask": torch.ones(batch, n_max, dtype=torch.bool),
        # Global
        "remaining_budget": torch.full((batch, 1), 3.0),
        "budget_fraction": torch.ones(batch, 1),
        "step_count": torch.zeros(batch, 1, dtype=torch.int32),
        "selected_fraction": torch.zeros(batch, 1),
        "current_pc": torch.full((batch, 1), 2e-5),
    }


class TestFlatObsExtractor:
    def test_output_shape(self):
        k = 5
        obs_space = _make_obs_space(k=k)
        extractor = FlatObsExtractor(obs_space)
        # v2 extractor: 3K + 5
        assert extractor.features_dim == 3 * k + 5

    def test_forward_produces_correct_shape(self):
        k = 5
        obs_space = _make_obs_space(k=k)
        extractor = FlatObsExtractor(obs_space)

        obs = _make_batch_obs(k=k, batch=2)
        features = extractor(obs)
        assert features.shape == (2, 3 * k + 5)

    def test_pc_scaling(self):
        k = 3
        obs_space = _make_obs_space(k=k, n_max=5)
        extractor = FlatObsExtractor(obs_space)

        # Build a minimal batch so we can read the last (PC) element
        obs = _make_batch_obs(k=k, n_max=5, batch=1)
        obs["current_pc"] = torch.tensor([[2e-5]])

        features = extractor(obs)
        # Last element is current_pc × 1e5
        pc_feature = features[0, -1].item()
        assert abs(pc_feature - 2.0) < 1e-4

    def test_area_scaling(self):
        """candidate_areas should be scaled by 1e-4 in the extractor."""
        k = 3
        obs_space = _make_obs_space(k=k, n_max=5)
        extractor = FlatObsExtractor(obs_space)

        obs = _make_batch_obs(k=k, n_max=5, batch=1)
        obs["candidate_areas"] = torch.tensor([[10000.0, 20000.0, 0.0]])

        features = extractor(obs)
        # Area block starts at index 2K (after action_mask + costs)
        area_start = 2 * k
        assert abs(features[0, area_start].item() - 1.0) < 1e-4
        assert abs(features[0, area_start + 1].item() - 2.0) < 1e-4
        assert features[0, area_start + 2].item() == 0.0


class TestEvalSummaryDataclass:
    def test_summary_structure(self):
        ep = EvalEpisodeResult(
            episode_return=1.0e-6,
            episode_steps=3,
            final_pc=2.5e-5,
            baseline_pc=2.0e-5,
            delta_pc_total=5e-6,
            selected_pu_ids=[1, 2, 3],
            step_rewards=[1e-6, 2e-6, 2e-6],
            step_pc_values=[2.1e-5, 2.3e-5, 2.5e-5],
        )
        summary = EvalSummary(
            n_episodes=1,
            mean_return=1e-6,
            mean_steps=3.0,
            mean_final_pc=2.5e-5,
            mean_delta_pc=5e-6,
            episodes=[ep],
        )
        assert summary.n_episodes == 1
        assert summary.episodes[0].episode_steps == 3
        assert summary.episodes[0].selected_pu_ids == [1, 2, 3]
