"""Unit tests for ``scripts/run_experiment.py``.

These tests do not invoke training. They exercise:
- the preset/override layering in ``build_config``,
- TensorBoard flag plumbing,
- that the resulting ``ExperimentConfig`` is consistent with the
  experiment-contract validators,
- that ``serialize_config`` round-trips the TensorBoard fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Make scripts/ importable as a flat module for testing.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_experiment as cli  # noqa: E402  (importable only after path setup)

from habconn.training.experiment import (  # noqa: E402
    ExperimentConfig,
    ExperimentPaths,
    resolve_tensorboard_log,
    serialize_config,
)


def _parse(*argv: str) -> "cli.argparse.Namespace":
    return cli.build_parser().parse_args(list(argv))


class TestPresetDefaults:
    def test_smoke_preset_defaults(self, tmp_path: Path) -> None:
        args = _parse("--preset", "smoke", "--output-root", str(tmp_path / "out"))
        config = cli.build_config(args)
        assert config.total_timesteps == 50
        assert config.checkpoint_freq == 16
        assert config.n_eval_episodes == 1
        assert config.run_name == "smoke_small_vector_001"

    def test_rehearsal_preset_defaults(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "rehearsal", "--output-root", str(tmp_path / "out")
        )
        config = cli.build_config(args)
        assert config.total_timesteps == 1024
        assert config.checkpoint_freq == 64
        assert config.n_eval_episodes == 3
        assert config.run_name == "rehearsal_small_vector_001"

    def test_default_preset_is_smoke(self, tmp_path: Path) -> None:
        # Sanity: parsing no preset flag gives the smoke preset.
        args = _parse("--output-root", str(tmp_path / "out"))
        config = cli.build_config(args)
        assert config.total_timesteps == 50


class TestOverrides:
    def test_cli_flags_override_preset(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "rehearsal",
            "--total-timesteps", "128",
            "--checkpoint-freq", "32",
            "--run-name", "custom_run",
            "--output-root", str(tmp_path / "out"),
        )
        config = cli.build_config(args)
        assert config.total_timesteps == 128
        assert config.checkpoint_freq == 32
        assert config.run_name == "custom_run"

    def test_package_root_paths_resolve(self, tmp_path: Path) -> None:
        args = _parse("--preset", "smoke", "--output-root", str(tmp_path / "out"))
        config = cli.build_config(args, pkg_root=tmp_path / "pkg")
        assert config.data_dir == tmp_path / "pkg" / "data" / "examples" / "small_vector_001"
        assert config.graphab_jar == tmp_path / "pkg" / "tools" / "graphab.jar"
        assert config.work_root == tmp_path / "pkg" / "tmp" / "training_runs"
        # Explicit --output-root wins over the package-root default.
        assert config.output_root == tmp_path / "out"

    def test_invalid_selection_metric_rejected(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "smoke",
            "--selection-metric", "made_up_metric",
            "--output-root", str(tmp_path / "out"),
        )
        with pytest.raises(ValueError):
            cli.build_config(args)

    def test_invalid_run_name_rejected(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "smoke",
            "--run-name", "../escape",
            "--output-root", str(tmp_path / "out"),
        )
        with pytest.raises(ValueError):
            cli.build_config(args)

    def test_invalid_n_envs_rejected(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "smoke",
            "--n-envs", "0",
            "--output-root", str(tmp_path / "out"),
        )
        with pytest.raises(ValueError):
            cli.build_config(args)


class TestTensorBoardFlag:
    def test_tensorboard_off_by_default(self, tmp_path: Path) -> None:
        args = _parse("--preset", "smoke", "--output-root", str(tmp_path / "out"))
        config = cli.build_config(args)
        assert config.enable_tensorboard is False
        assert config.tensorboard_log is None
        paths = ExperimentPaths.from_config(config)
        assert resolve_tensorboard_log(config, paths) is None

    def test_tensorboard_flag_uses_default_directory(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "smoke",
            "--run-name", "tb_run",
            "--output-root", str(tmp_path / "out"),
            "--tensorboard",
        )
        config = cli.build_config(args)
        assert config.enable_tensorboard is True
        assert config.tensorboard_log is None
        paths = ExperimentPaths.from_config(config)
        resolved = resolve_tensorboard_log(config, paths)
        assert resolved == paths.tensorboard_dir
        assert resolved == tmp_path / "out" / "tb_run" / "tensorboard"

    def test_explicit_tensorboard_log_overrides_default(self, tmp_path: Path) -> None:
        explicit = tmp_path / "elsewhere" / "tb"
        args = _parse(
            "--preset", "smoke",
            "--run-name", "tb_run",
            "--output-root", str(tmp_path / "out"),
            "--tensorboard-log", str(explicit),
        )
        config = cli.build_config(args)
        # --tensorboard-log implies enabling.
        assert config.enable_tensorboard is True
        assert config.tensorboard_log == explicit
        paths = ExperimentPaths.from_config(config)
        assert resolve_tensorboard_log(config, paths) == explicit

    def test_tb_log_name_override(self, tmp_path: Path) -> None:
        args = _parse(
            "--preset", "smoke",
            "--tensorboard",
            "--tb-log-name", "rehearsal_001",
            "--output-root", str(tmp_path / "out"),
        )
        config = cli.build_config(args)
        assert config.tb_log_name == "rehearsal_001"


class TestSerialization:
    def test_disabled_state_serializes(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            run_name="serialize_off",
            data_dir=tmp_path / "data",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "o",
        )
        payload = serialize_config(config)
        assert payload["enable_tensorboard"] is False
        assert payload["tensorboard_log"] is None
        assert payload["tb_log_name"] == "ppo"
        json.dumps(payload)  # Round-trippable.

    def test_enabled_state_round_trips(self, tmp_path: Path) -> None:
        explicit = tmp_path / "tb_logs"
        config = ExperimentConfig(
            run_name="serialize_on",
            data_dir=tmp_path / "data",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "o",
            enable_tensorboard=True,
            tensorboard_log=explicit,
            tb_log_name="custom",
        )
        payload = serialize_config(config)
        text = json.dumps(payload)
        reloaded = json.loads(text)
        assert reloaded["enable_tensorboard"] is True
        assert reloaded["tensorboard_log"] == str(explicit)
        assert reloaded["tb_log_name"] == "custom"

    def test_tb_log_name_must_be_non_empty_string(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(
                run_name="bad_tb_name",
                data_dir=tmp_path,
                graphab_jar=tmp_path / "g.jar",
                work_root=tmp_path / "w",
                output_root=tmp_path / "o",
                tb_log_name="   ",
            )

    def test_enable_tensorboard_must_be_bool(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            ExperimentConfig(
                run_name="bad_tb_enable",
                data_dir=tmp_path,
                graphab_jar=tmp_path / "g.jar",
                work_root=tmp_path / "w",
                output_root=tmp_path / "o",
                enable_tensorboard="yes",  # type: ignore[arg-type]
            )


class TestExperimentPathsTensorboardDir:
    def test_tensorboard_dir_layout(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            run_name="tb_layout",
            data_dir=tmp_path,
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "o",
        )
        paths = ExperimentPaths.from_config(config)
        assert paths.tensorboard_dir == tmp_path / "o" / "tb_layout" / "tensorboard"

    def test_ensure_does_not_create_tensorboard_dir(self, tmp_path: Path) -> None:
        # The contract directory should appear on disk only when
        # TensorBoard is actually requested at training time.
        config = ExperimentConfig(
            run_name="tb_no_create",
            data_dir=tmp_path,
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "o",
        )
        paths = ExperimentPaths.from_config(config)
        paths.ensure()
        assert paths.run_dir.is_dir()
        assert not paths.tensorboard_dir.exists()
