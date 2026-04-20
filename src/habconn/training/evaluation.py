"""Policy evaluation utilities.

Reusable helpers for running deterministic evaluation episodes on a
VectorHabitatEnv with a MaskablePPO (or compatible) policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalEpisodeResult:
    """Result of a single evaluation episode."""

    episode_return: float
    episode_steps: int
    final_pc: float
    baseline_pc: float
    delta_pc_total: float
    selected_pu_ids: list[int]
    step_rewards: list[float] = field(default_factory=list)
    step_pc_values: list[float] = field(default_factory=list)


@dataclass(slots=True)
class EvalSummary:
    """Summary of one or more evaluation episodes."""

    n_episodes: int
    mean_return: float
    mean_steps: float
    mean_final_pc: float
    mean_delta_pc: float
    episodes: list[EvalEpisodeResult]


def run_evaluation_episode(model, env, *, deterministic: bool = True) -> EvalEpisodeResult:
    """Run one deterministic evaluation episode.

    The env is reset inside this function. `model.predict` is called
    with the current action mask from `env.action_masks()` on each step.

    Parameters
    ----------
    model
        An SB3 / sb3-contrib model with a ``predict(obs, action_masks=...)`` API.
    env : VectorHabitatEnv
    deterministic : bool
        Whether to run the policy deterministically.

    Returns
    -------
    EvalEpisodeResult
    """
    obs, info = env.reset()
    baseline_pc = float(info.get("pc_value", 0.0))

    step_rewards: list[float] = []
    step_pcs: list[float] = []

    episode_return = 0.0
    done = False
    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))

        episode_return += float(reward)
        step_rewards.append(float(reward))
        step_pcs.append(float(info.get("pc_value", 0.0)))

        done = terminated or truncated

    final_pc = step_pcs[-1] if step_pcs else baseline_pc
    selected = list(info.get("selected_pu_ids", []))

    return EvalEpisodeResult(
        episode_return=episode_return,
        episode_steps=len(step_rewards),
        final_pc=final_pc,
        baseline_pc=baseline_pc,
        delta_pc_total=final_pc - baseline_pc,
        selected_pu_ids=selected,
        step_rewards=step_rewards,
        step_pc_values=step_pcs,
    )


def evaluate_policy(
    model,
    env,
    *,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> EvalSummary:
    """Run ``n_episodes`` evaluation episodes and summarize.

    Parameters
    ----------
    model
        An SB3 / sb3-contrib model.
    env : VectorHabitatEnv
    n_episodes : int
        Number of deterministic episodes to run.
    deterministic : bool

    Returns
    -------
    EvalSummary
    """
    episodes: list[EvalEpisodeResult] = []
    for _ in range(n_episodes):
        episodes.append(run_evaluation_episode(model, env, deterministic=deterministic))

    n = len(episodes)
    mean_return = sum(e.episode_return for e in episodes) / n
    mean_steps = sum(e.episode_steps for e in episodes) / n
    mean_final_pc = sum(e.final_pc for e in episodes) / n
    mean_delta_pc = sum(e.delta_pc_total for e in episodes) / n

    return EvalSummary(
        n_episodes=n,
        mean_return=mean_return,
        mean_steps=mean_steps,
        mean_final_pc=mean_final_pc,
        mean_delta_pc=mean_delta_pc,
        episodes=episodes,
    )
