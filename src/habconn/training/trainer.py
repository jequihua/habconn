"""Training entry points and orchestration utilities.

Provides `train_baseline()` which runs a masked-PPO training loop on a
VectorHabitatEnv using sb3-contrib's MaskablePPO with:
- FlatObsExtractor consuming the v2 observation
- EpisodeHistoryCallback writing a JSONL training history
- evaluate_policy() producing a structured post-training evaluation summary

Reward handling: raw delta-PC is kept as-is from the environment.
The reward magnitude is ~1e-6; MaskablePPO's advantage normalization
handles this without explicit scaling. Reward normalization is an
explicit non-goal for this milestone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO

from habconn.envs.vector_env import VectorHabitatEnv
from habconn.models.extractors.padded_mlp import FlatObsExtractor
from habconn.training.callbacks import EpisodeHistoryCallback
from habconn.training.evaluation import EvalSummary, evaluate_policy
from habconn.training.make_env import make_env


@dataclass
class BaselineConfig:
    """Configuration for the trainable baseline."""

    # Data paths
    data_dir: Path = Path("data/examples/small_vector_001")
    graphab_jar: Path = Path("tools/graphab.jar")
    work_root: Path = Path("tmp/training_runs")

    # Environment
    budget: int = 3
    k: int = 10
    seed: int = 42

    # Training
    total_timesteps: int = 50
    learning_rate: float = 3e-4
    n_steps: int = 8
    batch_size: int = 4
    n_epochs: int = 2
    gamma: float = 0.99

    # Evaluation
    n_eval_episodes: int = 1

    # Output
    output_dir: Path = Path("tmp/baseline_output")


def _eval_summary_to_dict(summary: EvalSummary) -> dict:
    return {
        "n_episodes": summary.n_episodes,
        "mean_return": summary.mean_return,
        "mean_steps": summary.mean_steps,
        "mean_final_pc": summary.mean_final_pc,
        "mean_delta_pc": summary.mean_delta_pc,
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
            for e in summary.episodes
        ],
    }


def train_baseline(
    config: Optional[BaselineConfig] = None,
    env: Optional[VectorHabitatEnv] = None,
) -> dict:
    """Run a masked-PPO training baseline and produce richer artifacts.

    Artifacts written to ``config.output_dir``:
    - ``baseline_model.zip`` — final SB3 model
    - ``history.jsonl`` — per-episode training history
    - ``baseline_summary.json`` — full training + evaluation summary

    Parameters
    ----------
    config : BaselineConfig or None
        Training configuration. Uses defaults if None.
    env : VectorHabitatEnv or None
        Pre-created environment. If None, creates one from config paths.

    Returns
    -------
    dict
        Summary dict (also written to baseline_summary.json).
    """
    if config is None:
        config = BaselineConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    if env is None:
        env = make_env(
            data_dir=config.data_dir,
            graphab_jar=config.graphab_jar,
            work_root=config.work_root,
            budget=config.budget,
            k=config.k,
            random_seed=config.seed,
        )

    # MaskablePPO with FlatObsExtractor
    policy_kwargs = {
        "features_extractor_class": FlatObsExtractor,
        "net_arch": [32, 32],
    }

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        verbose=1,
        seed=config.seed,
        policy_kwargs=policy_kwargs,
    )

    # Episode-history callback
    history_path = config.output_dir / "history.jsonl"
    history_cb = EpisodeHistoryCallback(output_path=history_path)

    # Train
    model.learn(total_timesteps=config.total_timesteps, callback=history_cb)

    # Structured evaluation
    eval_summary = evaluate_policy(
        model, env, n_episodes=config.n_eval_episodes, deterministic=True,
    )

    # Save model
    model_path = config.output_dir / "baseline_model"
    model.save(str(model_path))

    # Assemble summary
    summary = {
        "total_timesteps": config.total_timesteps,
        "n_training_episodes_logged": history_cb.episode_count,
        "history_path": str(history_path),
        "evaluation": _eval_summary_to_dict(eval_summary),
        "model_path": str(model_path),
        "config": {
            "data_dir": str(config.data_dir),
            "budget": config.budget,
            "k": config.k,
            "seed": config.seed,
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "n_eval_episodes": config.n_eval_episodes,
            "total_timesteps": config.total_timesteps,
        },
    }

    summary_path = config.output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
