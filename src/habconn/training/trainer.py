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
from stable_baselines3.common.callbacks import CallbackList

from habconn.envs.vector_env import VectorHabitatEnv
from habconn.models.extractors.padded_mlp import FlatObsExtractor
from habconn.training.baselines import run_evaluation_comparison
from habconn.training.callbacks import EpisodeHistoryCallback
from habconn.training.checkpointing import (
    CheckpointCallback,
    run_checkpoint_selection,
    validate_checkpoint_freq,
    validate_selection_metric,
    validate_selection_mode,
)
from habconn.training.deployment import run_deployment_export
from habconn.training.inspection import (
    observation_schema,
    summarize_observation_features,
    write_inspection_artifacts,
)
from habconn.training.evaluation import EvalSummary, evaluate_policy
from habconn.training.make_env import make_env
from habconn.training.vecenv import (
    make_vector_envs,
    worker_work_roots,
)


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

    # Vectorized training. ``n_envs == 1`` preserves the legacy single-env
    # path; ``n_envs > 1`` builds a worker-safe DummyVecEnv via
    # ``training/vecenv.py`` with isolated Graphab work roots per worker.
    n_envs: int = 1

    # Checkpointing + best-model selection. ``checkpoint_freq`` counts
    # SB3 callback invocations (see training/checkpointing.py for the
    # vec-env caveat). ``selection_metric`` is restricted to the
    # whitelist in training/checkpointing.py.
    checkpoint_freq: int = 16
    selection_metric: str = "mean_final_pc"
    selection_mode: str = "max"

    # Output
    output_dir: Path = Path("tmp/baseline_output")

    # Optional explicit artifact paths. When ``None`` (default) the trainer
    # falls back to legacy filenames inside ``output_dir``. When set (typically
    # by the experiment-contract layer), these override the defaults so the
    # trainer can write into a structured run directory.
    history_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    model_path: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    selection_dir: Optional[Path] = None
    best_model_path: Optional[Path] = None

    # TensorBoard. ``None`` disables TensorBoard logging entirely;
    # MaskablePPO is constructed without ``tensorboard_log`` so no
    # event file is created. When set, this path is passed directly to
    # ``MaskablePPO(..., tensorboard_log=...)`` and the directory is
    # created at training time.
    tensorboard_log: Optional[Path] = None
    tb_log_name: str = "ppo"


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

    if not isinstance(config.n_envs, int) or isinstance(config.n_envs, bool):
        raise TypeError(
            f"n_envs must be int, got {type(config.n_envs).__name__}"
        )
    if config.n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {config.n_envs}")
    if env is not None and config.n_envs > 1:
        raise ValueError(
            "Pre-created env is incompatible with n_envs > 1; pass workers "
            "via train_baseline(config=...) instead."
        )
    # Validate checkpointing + selection knobs at use site so callers
    # that bypass ExperimentConfig still get a clear error.
    validate_checkpoint_freq(config.checkpoint_freq)
    validate_selection_metric(config.selection_metric)
    validate_selection_mode(config.selection_mode)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve artifact paths. Optional overrides on the config let the
    # experiment-contract layer write into a structured run directory
    # without changing legacy single-file behavior.
    history_path = (
        Path(config.history_path)
        if config.history_path is not None
        else config.output_dir / "history.jsonl"
    )
    summary_path = (
        Path(config.summary_path)
        if config.summary_path is not None
        else config.output_dir / "baseline_summary.json"
    )
    model_path = (
        Path(config.model_path)
        if config.model_path is not None
        else config.output_dir / "baseline_model.zip"
    )
    checkpoints_dir = (
        Path(config.checkpoints_dir)
        if config.checkpoints_dir is not None
        else config.output_dir / "checkpoints"
    )
    selection_dir = (
        Path(config.selection_dir)
        if config.selection_dir is not None
        else config.output_dir / "selection"
    )
    best_model_path = (
        Path(config.best_model_path)
        if config.best_model_path is not None
        else model_path.parent / "best_model.zip"
    )
    for p in (history_path, summary_path, model_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    for d in (checkpoints_dir, selection_dir):
        d.mkdir(parents=True, exist_ok=True)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create training environment(s).
    #
    # n_envs == 1: keep the legacy single-env path so existing artifacts,
    # tests, and Graphab scratch layout remain identical.
    #
    # n_envs > 1: build a worker-safe DummyVecEnv via training/vecenv.py
    # with isolated Graphab work roots per worker. Evaluation always runs
    # on a separate single-env instance with its own scratch directory
    # so the evaluation path stays simple and deterministic.
    vec_env_type: Optional[str] = None
    worker_roots: Optional[list[Path]] = None
    eval_env: VectorHabitatEnv

    if config.n_envs == 1:
        if env is None:
            env = make_env(
                data_dir=config.data_dir,
                graphab_jar=config.graphab_jar,
                work_root=config.work_root,
                budget=config.budget,
                k=config.k,
                random_seed=config.seed,
            )
        train_env = env
        eval_env = env
    else:
        worker_roots = worker_work_roots(config.work_root, config.n_envs)
        train_env = make_vector_envs(
            n_envs=config.n_envs,
            data_dir=config.data_dir,
            graphab_jar=config.graphab_jar,
            work_root=config.work_root,
            budget=config.budget,
            k=config.k,
            base_seed=config.seed,
        )
        vec_env_type = type(train_env).__name__
        eval_env = make_env(
            data_dir=config.data_dir,
            graphab_jar=config.graphab_jar,
            work_root=config.work_root / "eval",
            budget=config.budget,
            k=config.k,
            random_seed=config.seed,
        )

    # MaskablePPO with FlatObsExtractor
    policy_kwargs = {
        "features_extractor_class": FlatObsExtractor,
        "net_arch": [32, 32],
    }

    # TensorBoard: create the log directory only when enabled, so the
    # contract directory does not appear on disk for runs that did not
    # request TensorBoard. SB3 expects a string path or None.
    tensorboard_log_path: Optional[Path] = (
        Path(config.tensorboard_log) if config.tensorboard_log is not None else None
    )
    if tensorboard_log_path is not None:
        tensorboard_log_path.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        verbose=1,
        seed=config.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=(
            str(tensorboard_log_path) if tensorboard_log_path is not None else None
        ),
    )

    # Episode-history callback + periodic checkpoint callback. SB3's
    # ``CallbackList`` runs each in order; both contribute to the
    # contract artifacts.
    history_cb = EpisodeHistoryCallback(output_path=history_path)
    checkpoint_cb = CheckpointCallback(
        save_dir=checkpoints_dir, save_freq=config.checkpoint_freq,
    )
    callbacks = CallbackList([history_cb, checkpoint_cb])

    # Train. ``tb_log_name`` controls the SB3 sub-run subdirectory
    # under ``tensorboard_log``. SB3 ignores it when ``tensorboard_log``
    # is None.
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name=config.tb_log_name,
    )

    # Structured evaluation on a single non-vectorized env so per-step
    # rewards and PC traces are unambiguous.
    eval_summary = evaluate_policy(
        model, eval_env, n_episodes=config.n_eval_episodes, deterministic=True,
    )

    # Save model. SB3 appends ``.zip`` if the path lacks the suffix; we
    # pass it without the suffix so behavior is identical regardless of
    # whether the resolved path ends in ``.zip``.
    model_save_stem = (
        model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    )
    model.save(str(model_save_stem))

    # Evaluation comparison: trained policy (already evaluated above) vs
    # three simple deterministic baselines. The comparison is written to
    # ``<output_dir>/evaluation/`` so the artifact stays inside the
    # experiment-contract layout. The trained_policy summary is reused
    # to avoid re-running the policy evaluation.
    evaluation_dir = config.output_dir / "evaluation"
    comparison = run_evaluation_comparison(
        env=eval_env,
        n_eval_episodes=config.n_eval_episodes,
        base_seed=config.seed,
        output_dir=evaluation_dir,
        trained_policy_summary=eval_summary,
        run_name=config.output_dir.name,
        budget=config.budget,
        k=config.k,
    )
    comparison_json_path = evaluation_dir / "comparison.json"
    comparison_csv_path = evaluation_dir / "comparison.csv"

    # Checkpoint candidate evaluation + best-model selection. This
    # re-evaluates the final model alongside every checkpoint so all
    # candidates share one symmetric evaluation surface; it does NOT
    # touch the trained-policy summary written into
    # baseline_summary.json or the comparison artifacts above.
    # Use the actual model timestep at end-of-training rather than the
    # requested ``config.total_timesteps``. SB3 PPO can overshoot by up
    # to ``n_steps - 1`` to finish a rollout, so the requested value is
    # not always the truth, and timestep participates in tie-breaking.
    final_actual_timestep = int(getattr(model, "num_timesteps", config.total_timesteps))

    # Hand the exact list of checkpoints saved during this training run
    # to the selector so stale files from a prior run with the same
    # ``run_name`` cannot contaminate selection.
    selection_result = run_checkpoint_selection(
        env=eval_env,
        final_model_path=model_path,
        final_model_timestep=final_actual_timestep,
        checkpoints_dir=checkpoints_dir,
        selection_dir=selection_dir,
        best_model_path=best_model_path,
        n_eval_episodes=config.n_eval_episodes,
        selection_metric=config.selection_metric,
        selection_mode=config.selection_mode,
        checkpoint_paths=checkpoint_cb.saved_paths,
    )
    checkpoint_evaluations_path = Path(selection_result["checkpoint_evaluations_path"])
    model_selection_path = Path(selection_result["model_selection_path"])

    # Deployment: load ``best_model.zip`` and run one deterministic
    # masked episode against the dedicated single-env eval. Selected
    # planning-unit geometries and a JSON summary land under
    # ``<output_dir>/deployment/``. The deployment env is the same
    # eval_env used above; ``run_deployment_export`` resets it before
    # the episode, so prior eval state cannot leak in.
    deployment_dir = config.output_dir / "deployment"
    deployment_result = run_deployment_export(
        model_path=best_model_path,
        env=eval_env,
        problem=eval_env.problem,
        output_dir=deployment_dir,
        run_name=config.output_dir.name,
        source_model_selection_path=model_selection_path,
        budget=config.budget,
        k=config.k,
        data_dir=config.data_dir,
    )
    deployment_summary_path = Path(deployment_result["deployment_summary_path"])
    selected_planning_units_gpkg_path = Path(
        deployment_result["selected_planning_units_gpkg_path"]
    )
    selected_planning_units_csv_path = Path(
        deployment_result["selected_planning_units_csv_path"]
    )

    # Inspection: write the v2 observation schema, a feature summary
    # over the initial deployment observation, and the per-step
    # action trace captured during the deployment episode above.
    # ``_initial_observation`` and ``_trace_steps`` are an internal
    # side channel from ``run_deployment_export`` (numpy arrays);
    # they are not part of any on-disk JSON contract.
    inspection_dir = config.output_dir / "inspection"
    initial_observation = deployment_result["_initial_observation"]
    trace_steps = deployment_result["_trace_steps"]
    schema_payload = observation_schema(initial_observation)
    feature_summary_payload = {
        "run_name": config.output_dir.name,
        "data_dir": str(config.data_dir),
        "budget": config.budget,
        "k": config.k,
        "n_planning_units": int(eval_env.problem.n_planning_units),
        "observation_version": "v2",
        "initial_observation": summarize_observation_features(initial_observation),
    }
    inspection_result = write_inspection_artifacts(
        inspection_dir,
        schema=schema_payload,
        feature_summary=feature_summary_payload,
        trace_steps=trace_steps,
        run_name=config.output_dir.name,
    )
    observation_schema_path = Path(inspection_result["observation_schema_path"])
    feature_summary_path = Path(inspection_result["feature_summary_path"])
    deployment_action_trace_json_path = Path(
        inspection_result["deployment_action_trace_json_path"]
    )
    deployment_action_trace_csv_path = Path(
        inspection_result["deployment_action_trace_csv_path"]
    )
    n_deployment_trace_steps = int(inspection_result["n_deployment_trace_steps"])
    n_deployment_trace_rows = int(inspection_result["n_deployment_trace_rows"])

    # Assemble summary
    summary = {
        "total_timesteps": config.total_timesteps,
        "n_training_episodes_logged": history_cb.episode_count,
        "history_path": str(history_path),
        "evaluation": _eval_summary_to_dict(eval_summary),
        "model_path": str(model_path),
        "evaluation_dir": str(evaluation_dir),
        "comparison_json_path": str(comparison_json_path),
        "comparison_csv_path": str(comparison_csv_path),
        "comparison_method_means": comparison["method_means"],
        "n_envs": config.n_envs,
        "vec_env_type": vec_env_type,
        "worker_work_roots": (
            [str(p) for p in worker_roots] if worker_roots is not None else None
        ),
        "checkpoints_dir": str(checkpoints_dir),
        "selection_dir": str(selection_dir),
        "best_model_path": str(best_model_path),
        "checkpoint_evaluations_path": str(checkpoint_evaluations_path),
        "model_selection_path": str(model_selection_path),
        "selected_candidate_id": selection_result["selected"]["candidate_id"],
        "selected_candidate_type": selection_result["selected"]["candidate_type"],
        "selected_candidate_timestep": selection_result["selected"]["timestep"],
        "selected_evaluation": selection_result["selected"]["evaluation"],
        "n_checkpoints_saved": len(checkpoint_cb.saved_steps),
        "deployment_dir": str(deployment_dir),
        "deployment_summary_path": str(deployment_summary_path),
        "selected_planning_units_gpkg_path": str(selected_planning_units_gpkg_path),
        "selected_planning_units_csv_path": str(selected_planning_units_csv_path),
        "deployment_selected_pu_ids": deployment_result["selected_pu_ids"],
        "deployment_final_pc": deployment_result["final_pc"],
        "deployment_delta_pc_total": deployment_result["delta_pc_total"],
        "inspection_dir": str(inspection_dir),
        "observation_schema_path": str(observation_schema_path),
        "feature_summary_path": str(feature_summary_path),
        "deployment_action_trace_json_path": str(deployment_action_trace_json_path),
        "deployment_action_trace_csv_path": str(deployment_action_trace_csv_path),
        "n_deployment_trace_steps": n_deployment_trace_steps,
        "n_deployment_trace_rows": n_deployment_trace_rows,
        "enable_tensorboard": tensorboard_log_path is not None,
        "tensorboard_log": (
            str(tensorboard_log_path) if tensorboard_log_path is not None else None
        ),
        "tb_log_name": config.tb_log_name,
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
            "n_envs": config.n_envs,
            "checkpoint_freq": config.checkpoint_freq,
            "selection_metric": config.selection_metric,
            "selection_mode": config.selection_mode,
            "enable_tensorboard": tensorboard_log_path is not None,
            "tensorboard_log": (
                str(tensorboard_log_path) if tensorboard_log_path is not None else None
            ),
            "tb_log_name": config.tb_log_name,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    summary["summary_path"] = str(summary_path)
    return summary
