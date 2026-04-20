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
