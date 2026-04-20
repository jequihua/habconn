"""Minimal feature extractor for the v2 observation dict.

Flattens the observation into a single feature vector for MLP-based
policies. Intentionally simple — a starting point for the first trainable
baseline, not a final architecture. For richer spatial or graph structure,
a dedicated encoder should be added later.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlatObsExtractor(BaseFeaturesExtractor):
    """Extracts a flat feature vector from the v2 Dict observation.

    Concatenates the following, in order:

    Action-level (3K features):
    - action_mask (K,) as float (0/1)
    - candidate_costs (K,) raw values (0 for padded slots)
    - candidate_areas (K,) scaled by ``area_scale`` (default 1e-4)

    Global (5 features):
    - remaining_budget (1,) raw
    - budget_fraction (1,) in [0, 1]
    - step_count (1,) raw as float
    - selected_fraction (1,) in [0, 1]
    - current_pc (1,) scaled by ``pc_scale`` (default 1e5)

    Total features_dim = 3K + 5.

    Scaling choices:
    - PC × 1e5: maps typical PC (~1e-5) to ~1.0
    - area × 1e-4: maps typical area (~1e4) to ~1.0

    Other fields (node-level arrays, candidate_ids, selected_mask,
    node_mask, node_costs, node_areas, eligibility_mask) are present in
    the observation but not consumed by this extractor. They are
    reserved for future encoders that handle variable-size nodes properly.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pc_scale: float = 1e5,
        area_scale: float = 1e-4,
    ) -> None:
        k = observation_space["action_mask"].shape[0]
        features_dim = 3 * k + 5
        super().__init__(observation_space, features_dim=features_dim)

        self._k = k
        self._pc_scale = pc_scale
        self._area_scale = area_scale

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        action_mask = observations["action_mask"].float()
        costs = observations["candidate_costs"]
        areas = observations["candidate_areas"] * self._area_scale

        budget = observations["remaining_budget"]
        budget_frac = observations["budget_fraction"]
        step = observations["step_count"].float()
        selected_frac = observations["selected_fraction"]
        pc = observations["current_pc"] * self._pc_scale

        return torch.cat(
            [action_mask, costs, areas, budget, budget_frac, step, selected_frac, pc],
            dim=-1,
        )
