"""Configurable single-landscape training rehearsal runner.

Thin CLI wrapper around ``habconn.training.experiment.run_experiment``.
All reusable training logic lives in the package; this script only
parses arguments, applies preset defaults, constructs an
``ExperimentConfig``, runs the experiment, and prints a compact
human-readable summary.

This is the single-landscape rehearsal entry point for both local and
HPC use. It does **not** implement transfer learning, new landscapes,
``SubprocVecEnv``, checkpoint resume, hyperparameter search, or reward
normalization.

Examples
--------

Compatibility-scale smoke run (matches the existing
``train_small_vector.py``)::

    .venv/Scripts/python scripts/run_experiment.py \\
        --preset smoke \\
        --run-name cli_smoke

Longer rehearsal with TensorBoard::

    .venv/Scripts/python scripts/run_experiment.py \\
        --preset rehearsal \\
        --run-name rehearsal_001 \\
        --output-root tmp/rehearsals \\
        --work-root tmp/rehearsal_runs \\
        --tensorboard
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from habconn.training.experiment import ExperimentConfig, run_experiment


PRESETS: dict[str, dict[str, Any]] = {
    # The smoke preset matches the compatibility scale used by
    # scripts/train_small_vector.py and HPC_HABCONN_SMOKE_TEST.md.
    "smoke": {
        "total_timesteps": 50,
        "checkpoint_freq": 16,
        "n_eval_episodes": 1,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 2,
    },
    # The rehearsal preset is meaningfully longer. It is still
    # conservative for the bundled small_vector_001 fixture; Graphab
    # CLI evaluation costs ~3-5 s per env step, so 1024 timesteps is
    # roughly an hour on a typical workstation.
    "rehearsal": {
        "total_timesteps": 1024,
        "checkpoint_freq": 64,
        "n_eval_episodes": 3,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 4,
    },
}


def _default_pkg_root() -> Path:
    """Resolve the package root (``08_pkg/habconn``) from this script."""
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-landscape MaskablePPO training rehearsal on "
            "small_vector_001. Writes the full experiment-contract "
            "artifact tree under <output_root>/<run_name>/."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="smoke",
        help=(
            "Apply preset defaults. 'smoke' = compatibility scale "
            "(50 timesteps); 'rehearsal' = longer learning run "
            "(1024 timesteps). Explicit flags override preset values. "
            "Default: smoke."
        ),
    )

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--graphab-jar", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)

    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)

    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)

    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--selection-mode", default=None)

    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help=(
            "Enable TensorBoard logging. When set without "
            "--tensorboard-log, logs land under "
            "<output_root>/<run_name>/tensorboard/."
        ),
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help=(
            "Override the default TensorBoard log directory. Implies "
            "--tensorboard."
        ),
    )
    parser.add_argument(
        "--tb-log-name",
        default=None,
        help="SB3 sub-run name appended under the TensorBoard log directory.",
    )

    return parser


def build_config(
    args: argparse.Namespace,
    *,
    pkg_root: Optional[Path] = None,
) -> ExperimentConfig:
    """Resolve preset defaults and CLI overrides into an ``ExperimentConfig``.

    Resolution order (later wins):
      1. ``ExperimentConfig`` field defaults,
      2. preset defaults from ``PRESETS[args.preset]``,
      3. package-root-relative defaults for paths,
      4. explicit CLI flags.

    A separate function so unit tests can exercise it without running
    Graphab.
    """
    root = pkg_root if pkg_root is not None else _default_pkg_root()
    preset = PRESETS[args.preset]

    defaults: dict[str, Any] = {
        "run_name": f"{args.preset}_small_vector_001",
        "data_dir": root / "data" / "examples" / "small_vector_001",
        "graphab_jar": root / "tools" / "graphab.jar",
        "work_root": root / "tmp" / "training_runs",
        "output_root": root / "tmp" / "experiments",
    }
    defaults.update(preset)

    # CLI flags override preset values when explicitly passed (i.e.
    # not None).
    overrides: dict[str, Any] = {
        "run_name": args.run_name,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "graphab_jar": args.graphab_jar,
        "work_root": args.work_root,
        "output_root": args.output_root,
        "budget": args.budget,
        "k": args.k,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "n_eval_episodes": args.n_eval_episodes,
        "n_envs": args.n_envs,
        "checkpoint_freq": args.checkpoint_freq,
        "selection_metric": args.selection_metric,
        "selection_mode": args.selection_mode,
    }
    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value

    # TensorBoard. Passing --tensorboard-log implies --tensorboard.
    enable_tb = bool(args.tensorboard or args.tensorboard_log is not None)
    if enable_tb:
        defaults["enable_tensorboard"] = True
        if args.tensorboard_log is not None:
            defaults["tensorboard_log"] = args.tensorboard_log
    if args.tb_log_name is not None:
        defaults["tb_log_name"] = args.tb_log_name

    return ExperimentConfig(**defaults)


def _print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=== Training Rehearsal Complete ===")
    print(f"  Run name              : {summary['run_name']}")
    print(f"  Run directory         : {summary['run_dir']}")
    print(f"  Total timesteps       : {summary['total_timesteps']}")
    print(f"  Training episodes     : {summary['n_training_episodes_logged']}")
    eval_summary = summary["evaluation"]
    print(f"  Eval episodes         : {eval_summary['n_episodes']}")
    print(f"  Eval mean return      : {eval_summary['mean_return']:.3e}")
    print(f"  Eval mean final PC    : {eval_summary['mean_final_pc']:.6e}")
    print(f"  Eval mean delta PC    : {eval_summary['mean_delta_pc']:.3e}")
    print(
        f"  Selected candidate    : {summary['selected_candidate_id']} "
        f"({summary['selected_candidate_type']}, "
        f"step {summary['selected_candidate_timestep']})"
    )
    print(f"  Deployment selected   : {summary['deployment_selected_pu_ids']}")
    print(f"  Deployment final PC   : {summary['deployment_final_pc']:.6e}")
    print(f"  Config path           : {summary['config_path']}")
    print(f"  Summary path          : {summary['summary_path']}")
    print(f"  Model path            : {summary['model_path']}")
    print(f"  Best model path       : {summary['best_model_path']}")
    print(f"  Comparison CSV        : {summary['comparison_csv_path']}")
    print(f"  Selection JSON        : {summary['model_selection_path']}")
    print(f"  Deployment summary    : {summary['deployment_summary_path']}")
    print(f"  Inspection schema     : {summary['observation_schema_path']}")
    if summary.get("enable_tensorboard"):
        tb_dir = summary.get("tensorboard_log")
        print(f"  TensorBoard log       : {tb_dir}")
        print(f"  TensorBoard launch    : tensorboard --logdir {tb_dir}")
    else:
        print("  TensorBoard           : disabled")


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    summary = run_experiment(config)
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
