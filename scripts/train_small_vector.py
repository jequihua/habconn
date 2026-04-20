"""Train a first small vector-action RL model.

Thin entry point. All reusable logic lives in habconn.training.

Usage:
    cd 08_pkg/habconn
    python scripts/train_small_vector.py
"""

from pathlib import Path

from habconn.training.trainer import BaselineConfig, train_baseline


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = BaselineConfig(
        data_dir=project_root / "data" / "examples" / "small_vector_001",
        graphab_jar=project_root / "tools" / "graphab.jar",
        work_root=project_root / "tmp" / "training_runs",
        output_dir=project_root / "tmp" / "baseline_output",
        budget=3,
        k=10,
        seed=42,
        total_timesteps=50,
        n_eval_episodes=1,
    )

    summary = train_baseline(config)

    eval_summary = summary["evaluation"]
    ep0 = eval_summary["episodes"][0]

    print("\n=== Baseline Training Complete ===")
    print(f"  Total timesteps       : {summary['total_timesteps']}")
    print(f"  Training episodes     : {summary['n_training_episodes_logged']}")
    print(f"  Eval episodes         : {eval_summary['n_episodes']}")
    print(f"  Eval mean return      : {eval_summary['mean_return']:.3e}")
    print(f"  Eval mean final PC    : {eval_summary['mean_final_pc']:.6e}")
    print(f"  Eval mean delta PC    : {eval_summary['mean_delta_pc']:.3e}")
    print(f"  Eval mean steps       : {eval_summary['mean_steps']:.1f}")
    print(f"  Ep0 baseline PC       : {ep0['baseline_pc']:.6e}")
    print(f"  Ep0 final PC          : {ep0['final_pc']:.6e}")
    print(f"  Ep0 selected PUs      : {ep0['selected_pu_ids']}")
    print(f"  Model path            : {summary['model_path']}")
    print(f"  History path          : {summary['history_path']}")
    print(f"  Summary path          : {config.output_dir / 'baseline_summary.json'}")


if __name__ == "__main__":
    main()
