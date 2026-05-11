"""Unit tests for the single-landscape experiment contract.

These tests do not require Graphab: they exercise config serialization,
the resolved on-disk layout, the metadata payload shape, and the
non-training portion of ``run_experiment`` via ``setup_experiment``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from habconn.training.experiment import (
    ExperimentConfig,
    ExperimentPaths,
    collect_metadata,
    serialize_config,
    setup_experiment,
)


def _make_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    base = dict(
        run_name="run_001",
        seed=7,
        data_dir=tmp_path / "data",
        graphab_jar=tmp_path / "graphab.jar",
        work_root=tmp_path / "scratch",
        output_root=tmp_path / "experiments",
        budget=2,
        k=5,
        total_timesteps=8,
        n_eval_episodes=1,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


class TestExperimentConfig:
    def test_default_construct(self):
        cfg = ExperimentConfig()
        assert cfg.run_name == "baseline"
        assert cfg.seed == 42
        assert cfg.budget == 3
        assert cfg.k == 10
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.output_root, Path)

    def test_string_paths_are_coerced(self, tmp_path):
        cfg = ExperimentConfig(
            run_name="run_a",
            data_dir=str(tmp_path / "data"),
            graphab_jar=str(tmp_path / "g.jar"),
            work_root=str(tmp_path / "work"),
            output_root=str(tmp_path / "out"),
        )
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.graphab_jar, Path)
        assert isinstance(cfg.work_root, Path)
        assert isinstance(cfg.output_root, Path)

    def test_empty_run_name_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            ExperimentConfig(run_name="", output_root=tmp_path)

    @pytest.mark.parametrize(
        "bad_name",
        [
            "   ",            # whitespace only
            ".",              # current directory
            "..",             # parent directory
            "../escaped",     # path traversal
            "..\\escaped",    # path traversal (Windows-style)
            "foo/bar",        # forward separator
            "foo\\bar",       # backslash separator
            "/abs/path",      # absolute-looking
            "foo..bar",       # contains '..' substring
            ".hidden",        # leading dot
            "-flag",          # leading hyphen
            "name with spaces",
            "name@with!punct",
        ],
    )
    def test_invalid_run_name_rejected(self, tmp_path, bad_name):
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(run_name=bad_name, output_root=tmp_path)

    def test_non_string_run_name_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            ExperimentConfig(run_name=123, output_root=tmp_path)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "good_name",
        [
            "baseline",
            "run_001",
            "run-2026-05-07",
            "v1.0",
            "Baseline_Small_Vector_001",
            "_internal",
            "0_first",
        ],
    )
    def test_valid_run_names_accepted(self, tmp_path, good_name):
        cfg = ExperimentConfig(run_name=good_name, output_root=tmp_path)
        assert cfg.run_name == good_name

    def test_default_n_envs_is_one(self, tmp_path):
        cfg = ExperimentConfig(run_name="ok", output_root=tmp_path)
        assert cfg.n_envs == 1

    @pytest.mark.parametrize("bad_n_envs", [0, -1, -10])
    def test_invalid_n_envs_rejected(self, tmp_path, bad_n_envs):
        with pytest.raises(ValueError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, n_envs=bad_n_envs,
            )

    def test_non_int_n_envs_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, n_envs=2.5,  # type: ignore[arg-type]
            )

    def test_bool_n_envs_rejected(self, tmp_path):
        # bool is a subclass of int; reject explicitly so True/False
        # cannot silently mean n_envs=1/0.
        with pytest.raises(TypeError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, n_envs=True,  # type: ignore[arg-type]
            )

    def test_n_envs_serializes(self, tmp_path):
        from habconn.training.experiment import serialize_config
        cfg = ExperimentConfig(run_name="ok", output_root=tmp_path, n_envs=4)
        payload = serialize_config(cfg)
        assert payload["n_envs"] == 4

    def test_checkpoint_defaults(self, tmp_path):
        cfg = ExperimentConfig(run_name="ok", output_root=tmp_path)
        assert cfg.checkpoint_freq == 16
        assert cfg.selection_metric == "mean_final_pc"
        assert cfg.selection_mode == "max"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_checkpoint_freq_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, checkpoint_freq=bad,
            )

    def test_bool_checkpoint_freq_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, checkpoint_freq=True,  # type: ignore[arg-type]
            )

    def test_invalid_selection_metric_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path,
                selection_metric="mean_pc",
            )

    def test_invalid_selection_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            ExperimentConfig(
                run_name="ok", output_root=tmp_path, selection_mode="min",
            )

    def test_checkpoint_knobs_serialize(self, tmp_path):
        from habconn.training.experiment import serialize_config
        cfg = ExperimentConfig(
            run_name="ok", output_root=tmp_path,
            checkpoint_freq=8,
            selection_metric="mean_delta_pc",
            selection_mode="max",
        )
        payload = serialize_config(cfg)
        assert payload["checkpoint_freq"] == 8
        assert payload["selection_metric"] == "mean_delta_pc"
        assert payload["selection_mode"] == "max"


class TestPathContainment:
    """Even when run_name is valid, the resolved run_dir must stay under output_root."""

    def test_run_dir_is_contained_under_output_root(self, tmp_path):
        cfg = ExperimentConfig(run_name="ok_name", output_root=tmp_path)
        paths = ExperimentPaths.from_config(cfg)
        # Resolved run dir is under resolved output root.
        paths.run_dir.resolve().relative_to(tmp_path.resolve())


class TestSerialization:
    def test_serialize_round_trip_via_json(self, tmp_path):
        cfg = _make_config(tmp_path)
        payload = serialize_config(cfg)
        # JSON-serializable
        text = json.dumps(payload)
        restored = json.loads(text)

        assert restored["run_name"] == cfg.run_name
        assert restored["seed"] == cfg.seed
        assert restored["budget"] == cfg.budget
        assert restored["k"] == cfg.k
        # Paths become strings
        assert restored["data_dir"] == str(cfg.data_dir)
        assert restored["output_root"] == str(cfg.output_root)
        # Numeric hyperparameters survive
        assert restored["learning_rate"] == cfg.learning_rate
        assert restored["total_timesteps"] == cfg.total_timesteps


class TestExperimentPaths:
    def test_layout(self, tmp_path):
        cfg = _make_config(tmp_path)
        paths = ExperimentPaths.from_config(cfg)

        run_dir = tmp_path / "experiments" / "run_001"
        assert paths.run_dir == run_dir
        assert paths.config_path == run_dir / "config.json"
        assert paths.metadata_path == run_dir / "metadata.json"
        assert paths.history_path == run_dir / "history.jsonl"
        assert paths.summary_path == run_dir / "baseline_summary.json"
        assert paths.models_dir == run_dir / "models"
        assert paths.model_path == run_dir / "models" / "final_model.zip"
        assert paths.evaluation_dir == run_dir / "evaluation"
        assert paths.comparison_json_path == run_dir / "evaluation" / "comparison.json"
        assert paths.comparison_csv_path == run_dir / "evaluation" / "comparison.csv"
        assert paths.best_model_path == run_dir / "models" / "best_model.zip"
        assert paths.checkpoints_dir == run_dir / "checkpoints"
        assert paths.selection_dir == run_dir / "selection"
        assert paths.checkpoint_evaluations_path == run_dir / "selection" / "checkpoint_evaluations.json"
        assert paths.model_selection_path == run_dir / "selection" / "model_selection.json"

    def test_ensure_creates_directories(self, tmp_path):
        cfg = _make_config(tmp_path)
        paths = ExperimentPaths.from_config(cfg)
        assert not paths.run_dir.exists()

        paths.ensure()

        assert paths.run_dir.is_dir()
        assert paths.models_dir.is_dir()
        assert paths.evaluation_dir.is_dir()
        assert paths.checkpoints_dir.is_dir()
        assert paths.selection_dir.is_dir()


class TestCollectMetadata:
    def test_required_keys_present(self, tmp_path):
        cfg = _make_config(tmp_path)
        meta = collect_metadata(cfg)

        for key in (
            "timestamp_utc",
            "run_name",
            "seed",
            "python_version",
            "platform",
            "habconn_version",
            "habconn_path",
            "git_commit",
            "dependencies",
            "paths",
        ):
            assert key in meta, f"missing metadata key: {key}"

        assert meta["run_name"] == cfg.run_name
        assert meta["seed"] == cfg.seed

        # python_version is a dotted string
        assert isinstance(meta["python_version"], str)
        assert "." in meta["python_version"]

        # dependencies dict contains expected probe keys (values may be None)
        deps = meta["dependencies"]
        for pkg in ("numpy", "torch", "gymnasium", "stable_baselines3", "sb3_contrib"):
            assert pkg in deps

        # paths recorded as strings
        assert meta["paths"]["data_dir"] == str(cfg.data_dir)
        assert meta["paths"]["graphab_jar"] == str(cfg.graphab_jar)
        assert meta["paths"]["output_root"] == str(cfg.output_root)

    def test_metadata_is_json_serializable(self, tmp_path):
        cfg = _make_config(tmp_path)
        meta = collect_metadata(cfg)
        # default=str fallback should not be needed for the standard shape
        json.dumps(meta)


class TestSetupExperiment:
    def test_writes_config_and_metadata(self, tmp_path):
        cfg = _make_config(tmp_path)
        paths = setup_experiment(cfg)

        # Layout matches what the contract advertises
        assert paths.run_dir.is_dir()
        assert paths.models_dir.is_dir()
        assert paths.config_path.exists()
        assert paths.metadata_path.exists()

        # config.json reflects the resolved config
        cfg_payload = json.loads(paths.config_path.read_text(encoding="utf-8"))
        assert cfg_payload["run_name"] == cfg.run_name
        assert cfg_payload["seed"] == cfg.seed
        assert cfg_payload["k"] == cfg.k
        assert cfg_payload["budget"] == cfg.budget

        # metadata.json contains the contract-required keys
        meta_payload = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
        assert meta_payload["run_name"] == cfg.run_name
        assert "python_version" in meta_payload
        assert "dependencies" in meta_payload

    def test_run_directories_are_isolated_per_run_name(self, tmp_path):
        cfg_a = _make_config(tmp_path, run_name="run_a")
        cfg_b = _make_config(tmp_path, run_name="run_b")
        paths_a = setup_experiment(cfg_a)
        paths_b = setup_experiment(cfg_b)

        assert paths_a.run_dir != paths_b.run_dir
        assert paths_a.config_path.exists()
        assert paths_b.config_path.exists()
