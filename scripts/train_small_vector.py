"""Train a first small vector-action RL model.

Thin entry point. All reusable logic lives in habconn.training.

Routes the existing single-env baseline through the experiment contract
defined in ``habconn.training.experiment``. Each invocation produces a
self-contained run directory under ``tmp/experiments/<run_name>/``.

Usage:
    cd 08_pkg/habconn
    python scripts/train_small_vector.py
"""

from pathlib import Path

from habconn.training.experiment import ExperimentConfig, run_experiment


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = ExperimentConfig(
        run_name="baseline_small_vector_001",
        seed=42,
        data_dir=project_root / "data" / "examples" / "small_vector_001",
        graphab_jar=project_root / "tools" / "graphab.jar",
        work_root=project_root / "tmp" / "training_runs",
        output_root=project_root / "tmp" / "experiments",
        budget=3,
        k=10,
        total_timesteps=50,
        n_eval_episodes=1,
    )

    summary = run_experiment(config)

    eval_summary = summary["evaluation"]
    ep0 = eval_summary["episodes"][0]

    print("\n=== Baseline Training Complete ===")
    print(f"  Run name              : {summary['run_name']}")
    print(f"  Run directory         : {summary['run_dir']}")
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
    print(f"  Config path           : {summary['config_path']}")
    print(f"  Metadata path         : {summary['metadata_path']}")
    print(f"  History path          : {summary['history_path']}")
    print(f"  Summary path          : {summary['summary_path']}")
    print(f"  Model path            : {summary['model_path']}")
    print(f"  Best model path       : {summary['best_model_path']}")
    print(f"  Checkpoints dir       : {summary['checkpoints_dir']}")
    print(f"  Checkpoints saved     : {summary['n_checkpoints_saved']}")
    print(f"  Selection JSON        : {summary['model_selection_path']}")
    print(f"  Candidate eval JSON   : {summary['checkpoint_evaluations_path']}")
    print(f"  Selected candidate    : {summary['selected_candidate_id']} "
          f"({summary['selected_candidate_type']}, step {summary['selected_candidate_timestep']})")
    print(f"  Deployment summary    : {summary['deployment_summary_path']}")
    print(f"  Selected PUs (GPKG)   : {summary['selected_planning_units_gpkg_path']}")
    print(f"  Selected PUs (CSV)    : {summary['selected_planning_units_csv_path']}")
    print(f"  Deployment selected   : {summary['deployment_selected_pu_ids']}")
    print(f"  Deployment final PC   : {summary['deployment_final_pc']:.6e}")
    print(f"  Observation schema    : {summary['observation_schema_path']}")
    print(f"  Feature summary       : {summary['feature_summary_path']}")
    print(f"  Action trace (JSON)   : {summary['deployment_action_trace_json_path']}")
    print(f"  Action trace (CSV)    : {summary['deployment_action_trace_csv_path']}")
    print(f"  Trace steps / rows    : {summary['n_deployment_trace_steps']} / {summary['n_deployment_trace_rows']}")


if __name__ == "__main__":
    main()
