"""Reward computation for habitat restoration environments.

The canonical v1 reward is raw delta-PC: the change in the Probability
of Connectivity metric between the pre-action and post-action states.

    reward = pc_after - pc_before

This keeps the reward grounded in the exact scientific objective with
no premature shaping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardResult:
    """Result of a single reward computation."""

    reward: float
    pc_before: float
    pc_after: float
    delta_pc: float


def compute_delta_pc_reward(
    pc_before: float,
    pc_after: float,
) -> RewardResult:
    """Compute the raw delta-PC reward.

    Parameters
    ----------
    pc_before : float
        PC value before the action (or at episode start for the first step).
    pc_after : float
        PC value after the action.

    Returns
    -------
    RewardResult
        Contains the reward (= delta_pc), both PC values, and the delta.
    """
    delta = pc_after - pc_before
    return RewardResult(
        reward=delta,
        pc_before=pc_before,
        pc_after=pc_after,
        delta_pc=delta,
    )
