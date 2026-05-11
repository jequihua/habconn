"""Simple, deterministic baselines for single-landscape evaluation.

Three reproducible baselines plus an evaluation-comparison writer:

- ``random_valid``: choose a valid candidate slot uniformly from the
  current ``env.action_masks()``, using an explicit seed.
- ``lowest_cost``: choose the valid slot with the lowest
  ``candidate_costs``.
- ``largest_area``: choose the valid slot with the largest
  ``candidate_areas``.

All three respect ``env.action_masks()`` and never select an invalid
slot. Deterministic baselines tie-break on the first valid slot among
ties (numpy ``argmin`` / ``argmax`` default).

The comparison writer ``run_evaluation_comparison()`` produces:

::

    <output_dir>/comparison.json
    <output_dir>/comparison.csv

where ``output_dir`` is typically ``run_dir/evaluation/`` so the
artifact stays inside the experiment-contract layout. The trained
policy summary is supplied by the caller (typically ``train_baseline``)
to avoid re-running the trained evaluation; baselines are evaluated
here against the same environment.

Scope notes:
- This is a single-landscape comparison. It is not an optimality proof
  and does not produce transfer-learning evidence.
- The baselines deliberately read no state beyond
  ``env.action_masks()`` and the action-level observation slices
  (``candidate_costs``, ``candidate_areas``); no lookahead, no
  optimization solver.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from habconn.training.evaluation import EvalEpisodeResult, EvalSummary


BASELINE_METHODS: tuple[str, ...] = ("random_valid", "lowest_cost", "largest_area")
ALL_METHODS: tuple[str, ...] = ("trained_policy",) + BASELINE_METHODS


# ---------------------------------------------------------------------------
# Action selectors
# ---------------------------------------------------------------------------


def _ensure_valid(action_mask: np.ndarray) -> None:
    if not np.any(action_mask):
        raise ValueError("Cannot select an action: action_mask has no valid slot")


def select_random_valid(
    action_mask: np.ndarray, *, rng: np.random.Generator
) -> int:
    """Return a uniformly random valid candidate slot.

    Reproducible via the supplied ``rng``.
    """
    _ensure_valid(action_mask)
    valid_indices = np.flatnonzero(action_mask)
    return int(rng.choice(valid_indices))


def select_lowest_cost(
    action_mask: np.ndarray, candidate_costs: np.ndarray
) -> int:
    """Return the valid slot with the lowest candidate cost.

    Tie-breaking: first occurrence of the minimum (numpy argmin default).
    Invalid slots are masked out by setting their cost to ``+inf``.
    """
    _ensure_valid(action_mask)
    masked = np.where(np.asarray(action_mask, dtype=bool), candidate_costs, np.inf)
    return int(np.argmin(masked))


def select_largest_area(
    action_mask: np.ndarray, candidate_areas: np.ndarray
) -> int:
    """Return the valid slot with the largest candidate area.

    Tie-breaking: first occurrence of the maximum (numpy argmax default).
    Invalid slots are masked out by setting their area to ``-inf``.
    """
    _ensure_valid(action_mask)
    masked = np.where(np.asarray(action_mask, dtype=bool), candidate_areas, -np.inf)
    return int(np.argmax(masked))


# ---------------------------------------------------------------------------
# Episode rollout for a given action_fn
# ---------------------------------------------------------------------------


def _run_baseline_episode(
    env, action_fn: Callable[..., int], *, rng: Optional[np.random.Generator]
) -> EvalEpisodeResult:
    """Run one episode driven by ``action_fn`` and capture the same fields
    as ``run_evaluation_episode`` does for the trained policy.
    """
    obs, info = env.reset()
    baseline_pc = float(info.get("pc_value", 0.0))

    step_rewards: list[float] = []
    step_pcs: list[float] = []
    episode_return = 0.0
    done = False

    while not done:
        masks = env.action_masks()
        action = action_fn(obs, masks, rng) if rng is not None else action_fn(obs, masks)
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


def _summarize(episodes: list[EvalEpisodeResult]) -> EvalSummary:
    n = len(episodes)
    if n == 0:
        raise ValueError("Cannot summarize zero episodes")
    return EvalSummary(
        n_episodes=n,
        mean_return=sum(e.episode_return for e in episodes) / n,
        mean_steps=sum(e.episode_steps for e in episodes) / n,
        mean_final_pc=sum(e.final_pc for e in episodes) / n,
        mean_delta_pc=sum(e.delta_pc_total for e in episodes) / n,
        episodes=episodes,
    )


# ---------------------------------------------------------------------------
# Per-baseline evaluation entry points
# ---------------------------------------------------------------------------


def evaluate_random_valid(env, *, n_episodes: int, base_seed: int) -> EvalSummary:
    """Evaluate the random-valid baseline.

    Each episode draws actions from ``np.random.default_rng(base_seed + ep_idx)``
    so the baseline is reproducible without sharing state across episodes.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    def action_fn(obs, mask, rng):
        return select_random_valid(mask, rng=rng)

    episodes: list[EvalEpisodeResult] = []
    for ep_idx in range(n_episodes):
        rng = np.random.default_rng(int(base_seed) + ep_idx)
        episodes.append(_run_baseline_episode(env, action_fn, rng=rng))
    return _summarize(episodes)


def evaluate_lowest_cost(env, *, n_episodes: int) -> EvalSummary:
    """Evaluate the lowest-cost baseline (deterministic)."""
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    def action_fn(obs, mask):
        return select_lowest_cost(mask, obs["candidate_costs"])

    episodes = [
        _run_baseline_episode(env, action_fn, rng=None) for _ in range(n_episodes)
    ]
    return _summarize(episodes)


def evaluate_largest_area(env, *, n_episodes: int) -> EvalSummary:
    """Evaluate the largest-area baseline (deterministic)."""
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    def action_fn(obs, mask):
        return select_largest_area(mask, obs["candidate_areas"])

    episodes = [
        _run_baseline_episode(env, action_fn, rng=None) for _ in range(n_episodes)
    ]
    return _summarize(episodes)


# ---------------------------------------------------------------------------
# Comparison writer
# ---------------------------------------------------------------------------


def _eval_summary_to_dict(s: EvalSummary) -> dict[str, Any]:
    return {
        "n_episodes": s.n_episodes,
        "mean_return": s.mean_return,
        "mean_steps": s.mean_steps,
        "mean_final_pc": s.mean_final_pc,
        "mean_delta_pc": s.mean_delta_pc,
        "episodes": [
            {
                "episode_return": e.episode_return,
                "episode_steps": e.episode_steps,
                "baseline_pc": e.baseline_pc,
                "final_pc": e.final_pc,
                "delta_pc_total": e.delta_pc_total,
                "selected_pu_ids": e.selected_pu_ids,
                "step_rewards": e.step_rewards,
                "step_pc_values": e.step_pc_values,
            }
            for e in s.episodes
        ],
    }


def _method_means(name: str, s: EvalSummary) -> dict[str, Any]:
    return {
        "method": name,
        "n_episodes": s.n_episodes,
        "mean_return": s.mean_return,
        "mean_final_pc": s.mean_final_pc,
        "mean_delta_pc": s.mean_delta_pc,
        "mean_steps": s.mean_steps,
    }


_CSV_COLUMNS: tuple[str, ...] = (
    "method",
    "n_episodes",
    "mean_return",
    "mean_final_pc",
    "mean_delta_pc",
    "mean_steps",
)


def write_comparison_artifacts(
    output_dir: Path,
    summaries: dict[str, EvalSummary],
    *,
    run_name: Optional[str],
    seed: Optional[int],
    budget: Optional[int],
    k: Optional[int],
    n_eval_episodes: int,
) -> dict[str, Any]:
    """Write ``comparison.json`` and ``comparison.csv`` under ``output_dir``.

    ``summaries`` maps method name → ``EvalSummary``. The keys are
    written to disk in the order they appear in ``summaries`` and
    the file name set is documented in the module docstring.

    Returns the assembled comparison dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_means = {name: _method_means(name, s) for name, s in summaries.items()}

    comparison = {
        "run_name": run_name,
        "seed": seed,
        "budget": budget,
        "k": k,
        "n_eval_episodes": n_eval_episodes,
        "method_means": method_means,
        "methods": {name: _eval_summary_to_dict(s) for name, s in summaries.items()},
    }

    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(_CSV_COLUMNS))
        for name, m in method_means.items():
            writer.writerow([m[col] for col in _CSV_COLUMNS])

    return comparison


def run_evaluation_comparison(
    *,
    env,
    n_eval_episodes: int,
    base_seed: int,
    output_dir: Path,
    trained_policy_summary: EvalSummary,
    run_name: Optional[str] = None,
    budget: Optional[int] = None,
    k: Optional[int] = None,
) -> dict[str, Any]:
    """Run baselines and write the comparison artifact.

    The trained-policy summary is supplied by the caller so we don't
    re-run the trained evaluation. Baselines are evaluated against
    ``env`` (which they will reset between episodes).
    """
    if n_eval_episodes < 1:
        raise ValueError(f"n_eval_episodes must be >= 1, got {n_eval_episodes}")

    summaries: dict[str, EvalSummary] = {
        "trained_policy": trained_policy_summary,
        "random_valid": evaluate_random_valid(
            env, n_episodes=n_eval_episodes, base_seed=base_seed
        ),
        "lowest_cost": evaluate_lowest_cost(env, n_episodes=n_eval_episodes),
        "largest_area": evaluate_largest_area(env, n_episodes=n_eval_episodes),
    }

    return write_comparison_artifacts(
        output_dir,
        summaries,
        run_name=run_name,
        seed=base_seed,
        budget=budget,
        k=k,
        n_eval_episodes=n_eval_episodes,
    )
