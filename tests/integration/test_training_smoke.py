"""Integration tests for the minimal trainable baseline.

These tests verify that the training path works end-to-end
on the bundled example landscape with a tiny training run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _skip_if_missing():
    root = _project_root()
    data_dir = root / "data" / "examples" / "small_vector_001"
    if not (data_dir / "candidates.shp").exists():
        pytest.skip("Example data not found")
    if not (root / "tools" / "graphab.jar").exists():
        pytest.skip("Graphab jar not found")


@pytest.mark.integration
class TestEnvFactory:
    def test_make_env_creates_valid_env(self):
        _skip_if_missing()
        from habconn.training.make_env import make_env

        root = _project_root()
        env = make_env(
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=root / "tmp" / "test_factory_runs",
            budget=2,
            k=5,
        )

        obs, info = env.reset()
        assert "action_mask" in obs
        assert obs["action_mask"].shape == (5,)
        assert math.isfinite(info["pc_value"])

    def test_make_env_action_masks_method(self):
        """Verify env exposes action_masks() for MaskablePPO."""
        _skip_if_missing()
        from habconn.training.make_env import make_env

        root = _project_root()
        env = make_env(
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=root / "tmp" / "test_masks_runs",
            budget=2,
            k=5,
        )

        env.reset()
        masks = env.action_masks()
        assert masks.dtype == bool
        assert masks.shape == (5,)
        assert masks.any()  # at least some valid actions


@pytest.mark.integration
class TestTrainingSmoke:
    def test_tiny_baseline_runs(self):
        """Run the smallest possible training baseline and verify it completes."""
        _skip_if_missing()
        from habconn.training.trainer import BaselineConfig, train_baseline

        root = _project_root()
        config = BaselineConfig(
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=root / "tmp" / "test_train_runs",
            output_dir=root / "tmp" / "test_train_output",
            budget=2,
            k=5,
            seed=42,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            n_eval_episodes=1,
        )

        summary = train_baseline(config)

        # Structural checks
        assert summary["total_timesteps"] == 16
        assert "evaluation" in summary
        assert "history_path" in summary

        eval_s = summary["evaluation"]
        assert eval_s["n_episodes"] == 1
        assert eval_s["mean_steps"] > 0
        assert math.isfinite(eval_s["mean_return"])
        assert math.isfinite(eval_s["mean_final_pc"])
        assert len(eval_s["episodes"]) == 1

        ep0 = eval_s["episodes"][0]
        assert len(ep0["selected_pu_ids"]) > 0
        assert len(ep0["step_rewards"]) == ep0["episode_steps"]
        assert len(ep0["step_pc_values"]) == ep0["episode_steps"]

    def test_history_jsonl_written(self):
        """Verify the episode-history callback writes a JSONL file."""
        _skip_if_missing()
        import json as _json
        from habconn.training.trainer import BaselineConfig, train_baseline

        root = _project_root()
        output_dir = root / "tmp" / "test_history_output"
        config = BaselineConfig(
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=root / "tmp" / "test_history_runs",
            output_dir=output_dir,
            budget=2,
            k=5,
            seed=42,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
        )

        summary = train_baseline(config)

        history_path = Path(summary["history_path"])
        assert history_path.exists()

        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        # Each line is valid JSON with expected keys
        rec = _json.loads(lines[0])
        assert {"step", "episode", "episode_reward", "episode_length"}.issubset(rec.keys())

    def test_vectorized_training_uses_isolated_worker_roots(self, tmp_path):
        """End-to-end: n_envs=2 produces isolated worker roots and works."""
        _skip_if_missing()
        import json as _json
        from habconn.training.experiment import ExperimentConfig, run_experiment

        root = _project_root()
        config = ExperimentConfig(
            run_name="vec_smoke",
            seed=42,
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=tmp_path / "training_runs",
            output_root=tmp_path / "experiments",
            budget=2,
            k=5,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            n_eval_episodes=1,
            n_envs=2,
        )

        summary = run_experiment(config)

        # Contract layout still holds.
        run_dir = tmp_path / "experiments" / "vec_smoke"
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "history.jsonl").exists()
        assert (run_dir / "baseline_summary.json").exists()
        assert (run_dir / "models" / "final_model.zip").exists()

        # n_envs threaded into summary + saved config.
        assert summary["n_envs"] == 2
        assert summary["vec_env_type"] == "DummyVecEnv"
        assert summary["worker_work_roots"] is not None
        assert len(summary["worker_work_roots"]) == 2

        cfg_payload = _json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg_payload["n_envs"] == 2

        on_disk = _json.loads((run_dir / "baseline_summary.json").read_text(encoding="utf-8"))
        assert on_disk["n_envs"] == 2
        assert on_disk["vec_env_type"] == "DummyVecEnv"
        assert on_disk["config"]["n_envs"] == 2

        # Worker scratch directories were actually created and isolated.
        worker_root_strs = summary["worker_work_roots"]
        worker_roots = [Path(p) for p in worker_root_strs]
        assert worker_roots[0] == tmp_path / "training_runs" / "worker_000"
        assert worker_roots[1] == tmp_path / "training_runs" / "worker_001"
        for r in worker_roots:
            assert r.exists()
        assert worker_roots[0] != worker_roots[1]

        # Eval still ran on the dedicated single-env scratch.
        eval_root = tmp_path / "training_runs" / "eval"
        assert eval_root.exists()

    def test_run_experiment_writes_checkpoints_and_selection(self, tmp_path):
        """End-to-end: run_experiment produces checkpoints + best model + selection JSONs."""
        _skip_if_missing()
        import json as _json

        from habconn.training.experiment import ExperimentConfig, run_experiment

        root = _project_root()
        config = ExperimentConfig(
            run_name="ckpt_smoke",
            seed=42,
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=tmp_path / "training_runs",
            output_root=tmp_path / "experiments",
            budget=2,
            k=5,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            n_eval_episodes=1,
            checkpoint_freq=8,
            selection_metric="mean_final_pc",
            selection_mode="max",
            enable_tensorboard=True,
        )

        summary = run_experiment(config)

        run_dir = tmp_path / "experiments" / "ckpt_smoke"

        # Existing contract artifacts still produced.
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "history.jsonl").exists()
        assert (run_dir / "baseline_summary.json").exists()
        assert (run_dir / "evaluation" / "comparison.json").exists()
        assert (run_dir / "evaluation" / "comparison.csv").exists()
        assert (run_dir / "models" / "final_model.zip").exists()

        # Checkpointing artifacts.
        checkpoints_dir = run_dir / "checkpoints"
        selection_dir = run_dir / "selection"
        assert checkpoints_dir.is_dir()
        assert selection_dir.is_dir()

        cps = sorted(checkpoints_dir.glob("checkpoint_*_steps.zip"))
        assert len(cps) >= 1, f"No checkpoints saved under {checkpoints_dir}"

        # Best-model artifact.
        best = run_dir / "models" / "best_model.zip"
        assert best.exists()

        # Selection JSONs.
        cp_eval_path = selection_dir / "checkpoint_evaluations.json"
        sel_path = selection_dir / "model_selection.json"
        assert cp_eval_path.exists()
        assert sel_path.exists()

        # Returned summary surfaces every new path.
        for key in (
            "checkpoints_dir", "selection_dir", "best_model_path",
            "checkpoint_evaluations_path", "model_selection_path",
            "selected_candidate_id", "selected_candidate_type",
            "selected_candidate_timestep", "selected_evaluation",
            "n_checkpoints_saved",
        ):
            assert key in summary, f"summary missing {key}"

        assert summary["best_model_path"] == str(best)
        assert summary["checkpoints_dir"] == str(checkpoints_dir)
        assert summary["selection_dir"] == str(selection_dir)
        assert summary["n_checkpoints_saved"] == len(cps)

        # Selection JSON declares metric, mode, selected candidate,
        # all candidate ids, and the best_model_path.
        selection = _json.loads(sel_path.read_text(encoding="utf-8"))
        assert selection["selection_metric"] == "mean_final_pc"
        assert selection["selection_mode"] == "max"
        assert selection["selected_candidate_id"] == summary["selected_candidate_id"]
        assert selection["best_model_path"] == str(best)
        assert "tie_break_rule" in selection
        # Every checkpoint plus the final model are candidates.
        assert len(selection["all_candidate_ids"]) == len(cps) + 1
        assert summary["selected_candidate_id"] in selection["all_candidate_ids"]

        # Per-candidate evaluations file: one record per candidate, one
        # of which is flagged ``selected``.
        cp_eval = _json.loads(cp_eval_path.read_text(encoding="utf-8"))
        assert cp_eval["selection_metric"] == "mean_final_pc"
        candidates = cp_eval["candidates"]
        assert len(candidates) == len(cps) + 1
        selected_count = sum(1 for c in candidates if c.get("selected"))
        assert selected_count == 1
        # Each record has the metric values that drove selection.
        for c in candidates:
            ev = c["evaluation"]
            assert {"mean_return", "mean_final_pc", "mean_delta_pc", "mean_steps"}.issubset(ev.keys())

        # Selected candidate's source model file actually exists on disk.
        selected_record = next(c for c in candidates if c["selected"])
        assert Path(selected_record["model_path"]).exists()

        # Final candidate timestep must reflect the actual end-of-training
        # ``model.num_timesteps`` (which can overshoot the requested
        # ``total_timesteps`` to finish a PPO rollout) rather than blindly
        # echoing ``config.total_timesteps``.
        final_record = next(c for c in candidates if c["candidate_type"] == "final")
        assert final_record["timestep"] >= config.total_timesteps

        # baseline_summary.json on disk also reflects the checkpoint-pass
        # state so reviewers can read everything from one file.
        on_disk = _json.loads((run_dir / "baseline_summary.json").read_text(encoding="utf-8"))
        assert on_disk["n_envs"] == 1
        assert on_disk["best_model_path"] == str(best)
        assert on_disk["selected_candidate_id"] == summary["selected_candidate_id"]
        assert on_disk["config"]["checkpoint_freq"] == 8
        assert on_disk["config"]["selection_metric"] == "mean_final_pc"
        assert on_disk["config"]["selection_mode"] == "max"

        # ---- Deployment artifacts (Stage 4 milestone 5) ----
        import geopandas as _gpd

        deployment_dir = run_dir / "deployment"
        deployment_summary_path = deployment_dir / "deployment_summary.json"
        gpkg_path = deployment_dir / "selected_planning_units.gpkg"
        csv_path = deployment_dir / "selected_planning_units.csv"

        assert deployment_dir.is_dir()
        assert deployment_summary_path.exists()
        assert gpkg_path.exists()
        assert csv_path.exists()

        # Returned summary surfaces every deployment path.
        for key in (
            "deployment_dir",
            "deployment_summary_path",
            "selected_planning_units_gpkg_path",
            "selected_planning_units_csv_path",
            "deployment_selected_pu_ids",
            "deployment_final_pc",
            "deployment_delta_pc_total",
        ):
            assert key in summary, f"summary missing {key}"
        assert summary["deployment_summary_path"] == str(deployment_summary_path)
        assert summary["selected_planning_units_gpkg_path"] == str(gpkg_path)
        assert summary["selected_planning_units_csv_path"] == str(csv_path)

        dep = _json.loads(deployment_summary_path.read_text(encoding="utf-8"))
        # Deployment must reference best_model.zip (the selected candidate),
        # not the final model directly.
        assert dep["model_path"] == str(best)
        # The selection JSON path is also recorded for traceability.
        assert dep["model_selection_path"] == str(sel_path)
        assert dep["deterministic"] is True
        assert dep["budget"] == config.budget
        assert dep["k"] == config.k
        # n_selected matches the list length.
        assert dep["n_selected"] == len(dep["selected_pu_ids"])

        # Deployment selected PUs match the exported CSV ids in selection order.
        csv_rows = csv_path.read_text(encoding="utf-8").splitlines()
        header = csv_rows[0].split(",")
        assert "selection_order" in header
        # The first CSV column is always the internal_id_column by construction.
        pu_col = header[0]
        rows = [r.split(",") for r in csv_rows[1:]]
        pu_idx = 0
        ord_idx = header.index("selection_order")
        if dep["selected_pu_ids"]:
            sorted_rows = sorted(rows, key=lambda r: int(r[ord_idx]))
            csv_ordered_ids = [int(r[pu_idx]) for r in sorted_rows]
            assert csv_ordered_ids == dep["selected_pu_ids"]

            # GeoPackage round-trip: the loaded frame contains the same
            # selected ids.
            gdf = _gpd.read_file(gpkg_path)
            gpkg_ids = set(int(x) for x in gdf[pu_col].tolist())
            assert gpkg_ids == set(dep["selected_pu_ids"])

        # baseline_summary.json mirrors deployment metrics for at-a-glance review.
        assert on_disk["deployment_summary_path"] == str(deployment_summary_path)
        assert on_disk["deployment_selected_pu_ids"] == dep["selected_pu_ids"]

        # ---- Inspection artifacts (Stage 4 milestone 6) ----
        import csv as _csv

        inspection_dir = run_dir / "inspection"
        observation_schema_path = inspection_dir / "observation_schema.json"
        feature_summary_path = inspection_dir / "feature_summary.json"
        trace_json_path = inspection_dir / "deployment_action_trace.json"
        trace_csv_path = inspection_dir / "deployment_action_trace.csv"

        assert inspection_dir.is_dir()
        assert observation_schema_path.exists()
        assert feature_summary_path.exists()
        assert trace_json_path.exists()
        assert trace_csv_path.exists()

        # Returned summary surfaces every inspection path + counts.
        for key in (
            "inspection_dir",
            "observation_schema_path",
            "feature_summary_path",
            "deployment_action_trace_json_path",
            "deployment_action_trace_csv_path",
            "n_deployment_trace_steps",
            "n_deployment_trace_rows",
        ):
            assert key in summary, f"summary missing {key}"
        assert summary["observation_schema_path"] == str(observation_schema_path)

        # Schema describes the v2 observation honestly: 14 keys, node-level
        # arrays marked unused by FlatObsExtractor.
        schema = _json.loads(observation_schema_path.read_text(encoding="utf-8"))
        assert schema["observation_version"] == "v2"
        assert schema["n_keys"] == 14
        node_records = [k for k in schema["keys"] if k["group"] == "node"]
        assert node_records
        for nk in node_records:
            assert nk["consumed_by_flat_extractor"] is False
        # FlatObsExtractor consumed set must include the action/global subset
        # but not node-level arrays.
        consumed = set(schema["flat_extractor_consumed_keys"])
        assert {"action_mask", "candidate_costs", "candidate_areas",
                "remaining_budget", "budget_fraction", "step_count",
                "selected_fraction", "current_pc"}.issubset(consumed)
        for nkey in ("selected_mask", "node_mask", "node_costs",
                      "node_areas", "eligibility_mask"):
            assert nkey not in consumed

        # Feature summary reports stats for every observation key.
        feat = _json.loads(feature_summary_path.read_text(encoding="utf-8"))
        assert feat["observation_version"] == "v2"
        assert feat["budget"] == config.budget
        assert feat["k"] == config.k
        assert set(feat["initial_observation"].keys()) == set(
            schema["keys"][i]["name"] for i in range(len(schema["keys"]))
        )

        # Deployment action trace JSON matches deployment summary on
        # selected_pu_ids and step count.
        trace = _json.loads(trace_json_path.read_text(encoding="utf-8"))
        assert trace["n_steps"] == summary["n_deployment_trace_steps"]
        assert trace["n_rows"] == summary["n_deployment_trace_rows"]
        if dep["selected_pu_ids"]:
            # The last step's selected_pu_ids_after must equal the
            # deployment summary's full selected list.
            assert trace["steps"][-1]["selected_pu_ids_after"] == dep["selected_pu_ids"]
            # Chosen pu ids in step order == deployment selected_pu_ids.
            chosen_in_order = [s["chosen_pu_id"] for s in trace["steps"]]
            assert chosen_in_order == dep["selected_pu_ids"]

        # CSV has one row per (step, slot).
        with trace_csv_path.open(encoding="utf-8") as f:
            csv_rows = list(_csv.DictReader(f))
        assert len(csv_rows) == summary["n_deployment_trace_rows"]
        if dep["selected_pu_ids"]:
            # Exactly one chosen row per step.
            for step_idx in range(1, trace["n_steps"] + 1):
                chosen = [r for r in csv_rows if int(r["step"]) == step_idx and r["chosen"] == "True"]
                assert len(chosen) == 1

        # ---- TensorBoard artifacts (training-rehearsal milestone) ----
        tensorboard_dir = run_dir / "tensorboard"
        # The contract directory must exist when TensorBoard was enabled
        # at config time. SB3 creates it on its first writer flush.
        assert tensorboard_dir.is_dir(), (
            f"TensorBoard dir was not created at {tensorboard_dir}"
        )
        # Surfaced fields on both the returned summary and the on-disk
        # baseline_summary.json must agree, and tensorboard_log must
        # equal the contract tensorboard_dir when no override was
        # supplied.
        assert summary["enable_tensorboard"] is True
        assert summary["tensorboard_log"] == str(tensorboard_dir)
        assert summary["tb_log_name"] == "ppo"
        # baseline_summary.json must mirror the returned summary on
        # all three TensorBoard fields, both at top level and inside
        # the embedded ``config`` block.
        assert on_disk["enable_tensorboard"] is True
        assert on_disk["tensorboard_log"] == str(tensorboard_dir)
        assert on_disk["tb_log_name"] == "ppo"
        assert on_disk["config"]["enable_tensorboard"] is True
        assert on_disk["config"]["tensorboard_log"] == str(tensorboard_dir)
        assert on_disk["config"]["tb_log_name"] == "ppo"
        # config.json (the resolved ExperimentConfig) records the
        # enabled state. ``tensorboard_log`` stays ``None`` because no
        # explicit override was supplied; the resolved directory only
        # shows up in baseline_summary.json above.
        config_on_disk = _json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        assert config_on_disk["enable_tensorboard"] is True
        assert config_on_disk["tensorboard_log"] is None
        assert config_on_disk["tb_log_name"] == "ppo"

    def test_run_experiment_writes_evaluation_comparison(self, tmp_path):
        """End-to-end: run_experiment produces the comparison artifacts."""
        _skip_if_missing()
        import csv as _csv
        import json as _json
        import math as _math

        from habconn.training.experiment import ExperimentConfig, run_experiment

        root = _project_root()
        config = ExperimentConfig(
            run_name="eval_smoke",
            seed=42,
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=tmp_path / "training_runs",
            output_root=tmp_path / "experiments",
            budget=2,
            k=5,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            n_eval_episodes=1,
        )

        summary = run_experiment(config)

        run_dir = tmp_path / "experiments" / "eval_smoke"
        eval_dir = run_dir / "evaluation"

        # Existing contract artifacts still exist.
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "history.jsonl").exists()
        assert (run_dir / "baseline_summary.json").exists()
        assert (run_dir / "models" / "final_model.zip").exists()

        # New evaluation comparison artifacts.
        assert eval_dir.is_dir()
        comparison_json = eval_dir / "comparison.json"
        comparison_csv = eval_dir / "comparison.csv"
        assert comparison_json.exists()
        assert comparison_csv.exists()

        # Summary surfaces the comparison artifact paths.
        assert summary["comparison_json_path"] == str(comparison_json)
        assert summary["comparison_csv_path"] == str(comparison_csv)
        assert summary["evaluation_dir"] == str(eval_dir)

        # comparison.json contains all four methods with finite means.
        comp = _json.loads(comparison_json.read_text(encoding="utf-8"))
        assert comp["run_name"] == "eval_smoke"
        assert comp["seed"] == 42
        assert comp["n_eval_episodes"] == 1
        method_names = {"trained_policy", "random_valid", "lowest_cost", "largest_area"}
        assert set(comp["methods"].keys()) == method_names
        assert set(comp["method_means"].keys()) == method_names
        for name in method_names:
            m = comp["method_means"][name]
            assert _math.isfinite(m["mean_final_pc"])
            assert _math.isfinite(m["mean_delta_pc"])
            ep0 = comp["methods"][name]["episodes"][0]
            # Each method either selected at least one PU or terminated
            # before any selection (an explicitly accepted edge case).
            assert isinstance(ep0["selected_pu_ids"], list)

        # CSV contains exactly the four method rows.
        with comparison_csv.open(encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert {r["method"] for r in rows} == method_names
        for r in rows:
            float(r["mean_final_pc"])
            float(r["mean_delta_pc"])

        # Comparison method means also surface inside the saved
        # baseline_summary so reviewers can see both at once.
        on_disk = _json.loads((run_dir / "baseline_summary.json").read_text(encoding="utf-8"))
        assert "comparison_method_means" in on_disk
        assert set(on_disk["comparison_method_means"].keys()) == method_names

    def test_run_experiment_produces_contract_layout(self, tmp_path):
        """End-to-end: run_experiment writes the contract-defined artifacts."""
        _skip_if_missing()
        import json as _json
        from habconn.training.experiment import ExperimentConfig, run_experiment

        root = _project_root()
        config = ExperimentConfig(
            run_name="smoke_contract",
            seed=42,
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=tmp_path / "training_runs",
            output_root=tmp_path / "experiments",
            budget=2,
            k=5,
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            n_eval_episodes=1,
        )

        summary = run_experiment(config)

        run_dir = tmp_path / "experiments" / "smoke_contract"
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "history.jsonl").exists()
        assert (run_dir / "baseline_summary.json").exists()
        assert (run_dir / "models" / "final_model.zip").exists()

        # The summary returned in-memory matches what is on disk
        on_disk = _json.loads((run_dir / "baseline_summary.json").read_text(encoding="utf-8"))
        assert on_disk["total_timesteps"] == 16
        assert "evaluation" in on_disk

        # Convenience fields surfaced by run_experiment
        assert summary["run_name"] == "smoke_contract"
        assert summary["run_dir"] == str(run_dir)
        assert summary["config_path"] == str(run_dir / "config.json")
        assert summary["metadata_path"] == str(run_dir / "metadata.json")

        # Saved config faithfully captures inputs
        cfg_payload = _json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg_payload["run_name"] == "smoke_contract"
        assert cfg_payload["seed"] == 42
        assert cfg_payload["budget"] == 2
        assert cfg_payload["k"] == 5
        assert cfg_payload["total_timesteps"] == 16

    def test_masked_actions_prevent_invalid_slots(self):
        """Verify that MaskablePPO never selects a padded action slot."""
        _skip_if_missing()
        from sb3_contrib import MaskablePPO
        from habconn.training.make_env import make_env
        from habconn.models.extractors.padded_mlp import FlatObsExtractor

        root = _project_root()
        env = make_env(
            data_dir=root / "data" / "examples" / "small_vector_001",
            graphab_jar=root / "tools" / "graphab.jar",
            work_root=root / "tmp" / "test_masking_runs",
            budget=2,
            k=10,
        )

        model = MaskablePPO(
            "MultiInputPolicy", env,
            n_steps=8, batch_size=4, n_epochs=1,
            seed=42,
            policy_kwargs={"features_extractor_class": FlatObsExtractor, "net_arch": [16]},
        )

        # Run a few predict steps and check masks are respected
        obs, info = env.reset()
        for _ in range(3):
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            # Action must be in a valid slot
            assert masks[action], f"Model selected invalid slot {action}"

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
