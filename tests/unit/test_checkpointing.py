"""Unit tests for checkpointing + best-model selection helpers.

These tests exercise the parts of ``training/checkpointing.py`` that
do not require Graphab or a trained model: validators, filename
parsing, checkpoint discovery, and the deterministic selection rule
including tie-breaking.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from habconn.training.checkpointing import (
    CHECKPOINT_FILENAME_PATTERN,
    TIE_BREAK_RULE,
    VALID_SELECTION_METRICS,
    VALID_SELECTION_MODES,
    checkpoint_filename,
    discover_checkpoints,
    select_best_candidate,
    step_from_checkpoint_path,
    validate_checkpoint_freq,
    validate_selection_metric,
    validate_selection_mode,
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidateCheckpointFreq:
    @pytest.mark.parametrize("v", [1, 2, 16, 1000])
    def test_positive_int_passes(self, v):
        assert validate_checkpoint_freq(v) == v

    @pytest.mark.parametrize("v", [0, -1, -16])
    def test_non_positive_rejected(self, v):
        with pytest.raises(ValueError):
            validate_checkpoint_freq(v)

    @pytest.mark.parametrize("v", [True, False])
    def test_bool_rejected(self, v):
        with pytest.raises(TypeError):
            validate_checkpoint_freq(v)

    @pytest.mark.parametrize("v", [2.5, "8", None, [8]])
    def test_non_int_rejected(self, v):
        with pytest.raises(TypeError):
            validate_checkpoint_freq(v)


class TestValidateSelectionMetric:
    @pytest.mark.parametrize("m", VALID_SELECTION_METRICS)
    def test_whitelist_passes(self, m):
        assert validate_selection_metric(m) == m

    @pytest.mark.parametrize(
        "m",
        ["mean_pc", "mean_episode_return", "max_final_pc", "", "MEAN_FINAL_PC"],
    )
    def test_other_strings_rejected(self, m):
        with pytest.raises(ValueError):
            validate_selection_metric(m)

    @pytest.mark.parametrize("m", [42, None, ["mean_final_pc"]])
    def test_non_string_rejected(self, m):
        with pytest.raises(TypeError):
            validate_selection_metric(m)


class TestValidateSelectionMode:
    def test_max_passes(self):
        assert validate_selection_mode("max") == "max"

    @pytest.mark.parametrize("m", ["min", "MAX", "argmax", ""])
    def test_other_strings_rejected(self, m):
        with pytest.raises(ValueError):
            validate_selection_mode(m)

    @pytest.mark.parametrize("m", [1, None, ["max"]])
    def test_non_string_rejected(self, m):
        with pytest.raises(TypeError):
            validate_selection_mode(m)


# ---------------------------------------------------------------------------
# Filename layout
# ---------------------------------------------------------------------------


class TestCheckpointFilenamePattern:
    def test_pattern_zero_padded_six_digits(self):
        assert checkpoint_filename(0) == "checkpoint_000000_steps.zip"
        assert checkpoint_filename(16) == "checkpoint_000016_steps.zip"
        assert checkpoint_filename(123_456) == "checkpoint_123456_steps.zip"

    def test_pattern_round_trip(self):
        for step in (0, 1, 16, 99, 12345, 999_999):
            name = checkpoint_filename(step)
            assert step_from_checkpoint_path(Path(name)) == step

    def test_negative_step_rejected(self):
        with pytest.raises(ValueError):
            checkpoint_filename(-1)

    def test_step_from_unknown_filename_rejected(self):
        with pytest.raises(ValueError):
            step_from_checkpoint_path(Path("rl_model_16_steps.zip"))
        with pytest.raises(ValueError):
            step_from_checkpoint_path(Path("checkpoint.zip"))


class TestDiscoverCheckpoints:
    def test_empty_dir(self, tmp_path):
        assert discover_checkpoints(tmp_path) == []

    def test_missing_dir(self, tmp_path):
        assert discover_checkpoints(tmp_path / "missing") == []

    def test_sorted_by_step(self, tmp_path):
        # Create out-of-order checkpoint files plus a stray file.
        for step in (32, 8, 24, 16):
            (tmp_path / checkpoint_filename(step)).touch()
        (tmp_path / "stray.txt").touch()
        (tmp_path / "rl_model_99_steps.zip").touch()
        (tmp_path / "checkpoint.zip").touch()

        paths = discover_checkpoints(tmp_path)
        assert [step_from_checkpoint_path(p) for p in paths] == [8, 16, 24, 32]


# ---------------------------------------------------------------------------
# Selection rule + tie-breaking
# ---------------------------------------------------------------------------


def _candidate(
    *, cid, kind, timestep, mean_final_pc=0.0, mean_delta_pc=0.0,
    mean_return=0.0, mean_steps=3.0, model_path="x.zip",
):
    return {
        "candidate_id": cid,
        "model_path": model_path,
        "candidate_type": kind,
        "timestep": timestep,
        "evaluation": {
            "n_episodes": 1,
            "mean_return": mean_return,
            "mean_final_pc": mean_final_pc,
            "mean_delta_pc": mean_delta_pc,
            "mean_steps": mean_steps,
        },
    }


class TestSelectBestCandidate:
    def test_picks_highest_metric(self):
        cs = [
            _candidate(cid="cp_8", kind="checkpoint", timestep=8, mean_final_pc=1.0),
            _candidate(cid="cp_16", kind="checkpoint", timestep=16, mean_final_pc=3.0),
            _candidate(cid="final", kind="final", timestep=20, mean_final_pc=2.0),
        ]
        chosen = select_best_candidate(cs, "mean_final_pc", "max")
        assert chosen["candidate_id"] == "cp_16"

    def test_ties_break_by_later_timestep(self):
        # Same metric value: later timestep should win.
        cs = [
            _candidate(cid="cp_8", kind="checkpoint", timestep=8, mean_final_pc=2.0),
            _candidate(cid="cp_16", kind="checkpoint", timestep=16, mean_final_pc=2.0),
        ]
        chosen = select_best_candidate(cs, "mean_final_pc", "max")
        assert chosen["candidate_id"] == "cp_16"

    def test_ties_break_final_over_checkpoint_at_equal_timestep(self):
        # Same metric AND same timestep: final wins.
        cs = [
            _candidate(cid="cp_16", kind="checkpoint", timestep=16, mean_final_pc=2.0),
            _candidate(cid="final", kind="final", timestep=16, mean_final_pc=2.0),
        ]
        chosen = select_best_candidate(cs, "mean_final_pc", "max")
        assert chosen["candidate_id"] == "final"

    def test_works_for_all_whitelisted_metrics(self):
        cs = [
            _candidate(
                cid="a", kind="checkpoint", timestep=4,
                mean_final_pc=0.5, mean_delta_pc=0.1, mean_return=10.0,
            ),
            _candidate(
                cid="b", kind="checkpoint", timestep=8,
                mean_final_pc=0.3, mean_delta_pc=0.4, mean_return=5.0,
            ),
        ]
        assert select_best_candidate(cs, "mean_final_pc", "max")["candidate_id"] == "a"
        assert select_best_candidate(cs, "mean_delta_pc", "max")["candidate_id"] == "b"
        assert select_best_candidate(cs, "mean_return", "max")["candidate_id"] == "a"

    def test_invalid_metric_rejected(self):
        cs = [_candidate(cid="a", kind="final", timestep=1)]
        with pytest.raises(ValueError):
            select_best_candidate(cs, "mean_pc", "max")

    def test_invalid_mode_rejected(self):
        cs = [_candidate(cid="a", kind="final", timestep=1)]
        with pytest.raises(ValueError):
            select_best_candidate(cs, "mean_final_pc", "min")

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError):
            select_best_candidate([], "mean_final_pc", "max")

    def test_tie_break_rule_documented(self):
        # The rule string must reference the two tie-break stages so it
        # stays in sync with the selection function.
        assert "timestep" in TIE_BREAK_RULE.lower()
        assert "final" in TIE_BREAK_RULE.lower()


# ---------------------------------------------------------------------------
# Selection JSON shape (sanity check on serializability)
# ---------------------------------------------------------------------------


class TestSelectionPayloadSerializable:
    """A representative selection payload must be JSON-serializable.

    The trainer writes the payload via ``json.dump(..., default=str)``;
    this test asserts that a typical record needs no fallback at all.
    """

    def test_round_trip(self, tmp_path):
        cs = [
            _candidate(cid="cp_8", kind="checkpoint", timestep=8, mean_final_pc=1.0),
            _candidate(cid="cp_16", kind="checkpoint", timestep=16, mean_final_pc=3.0),
            _candidate(cid="final", kind="final", timestep=16, mean_final_pc=3.0),
        ]
        chosen = select_best_candidate(cs, "mean_final_pc", "max")
        for c in cs:
            c["selected"] = c["candidate_id"] == chosen["candidate_id"]

        payload = {
            "selection_metric": "mean_final_pc",
            "selection_mode": "max",
            "tie_break_rule": TIE_BREAK_RULE,
            "selected_candidate_id": chosen["candidate_id"],
            "selected_candidate_type": chosen["candidate_type"],
            "selected_candidate_timestep": chosen["timestep"],
            "selected_model_path": chosen["model_path"],
            "best_model_path": str(tmp_path / "best_model.zip"),
            "selected_evaluation": chosen["evaluation"],
            "all_candidate_ids": [c["candidate_id"] for c in cs],
        }
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["selected_candidate_id"] == chosen["candidate_id"]
        assert restored["all_candidate_ids"] == ["cp_8", "cp_16", "final"]


# ---------------------------------------------------------------------------
# Whitelists are exposed
# ---------------------------------------------------------------------------


def test_metric_whitelist_constant():
    assert VALID_SELECTION_METRICS == ("mean_final_pc", "mean_delta_pc", "mean_return")


def test_mode_whitelist_constant():
    assert VALID_SELECTION_MODES == ("max",)


def test_filename_pattern_constant():
    # The pattern should still be sortable and parseable.
    assert "{step:06d}" in CHECKPOINT_FILENAME_PATTERN


# ---------------------------------------------------------------------------
# Closure-pass regressions
# ---------------------------------------------------------------------------


class TestRunCheckpointSelectionStaleIsolation:
    """A stale checkpoint left over from a prior run must not be considered
    when the trainer hands ``run_checkpoint_selection`` an explicit list of
    current-run checkpoint paths."""

    def _patch_eval(self, monkeypatch, scores: dict[str, float]):
        """Replace the real model-load-and-evaluate with a stub that
        returns a deterministic mean_final_pc keyed by file stem."""
        from habconn.training import checkpointing as cp_mod
        from habconn.training.evaluation import EvalEpisodeResult, EvalSummary

        def _stub(model_path, env, *, n_episodes):
            stem = Path(model_path).stem
            score = scores[stem]
            ep = EvalEpisodeResult(
                episode_return=score, episode_steps=1,
                final_pc=score, baseline_pc=0.0, delta_pc_total=score,
                selected_pu_ids=[1],
                step_rewards=[score], step_pc_values=[score],
            )
            return EvalSummary(
                n_episodes=1,
                mean_return=score,
                mean_steps=1.0,
                mean_final_pc=score,
                mean_delta_pc=score,
                episodes=[ep],
            )
        monkeypatch.setattr(cp_mod, "_evaluate_model_at_path", _stub)

    def test_explicit_checkpoint_paths_exclude_stale_files(self, tmp_path, monkeypatch):
        from habconn.training.checkpointing import (
            checkpoint_filename, run_checkpoint_selection,
        )

        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        sel_dir = run_dir / "selection"
        models_dir = run_dir / "models"
        for d in (ckpt_dir, sel_dir, models_dir):
            d.mkdir(parents=True)

        # Current-run checkpoints (saved during this hypothetical run).
        current_paths = []
        for step in (8, 16):
            p = ckpt_dir / checkpoint_filename(step)
            p.write_bytes(b"")
            current_paths.append(p)

        # Stale checkpoint left over from a prior run with the SAME run_name.
        # Give it a metric value that would otherwise win selection so the
        # test fails loudly if the stale file leaks in.
        stale = ckpt_dir / checkpoint_filename(99)
        stale.write_bytes(b"")

        # Final model (from the current run).
        final = models_dir / "final_model.zip"
        final.write_bytes(b"")
        best = models_dir / "best_model.zip"

        # Stale would beat current candidates if it leaked in.
        self._patch_eval(monkeypatch, scores={
            checkpoint_filename(8).removesuffix(".zip"): 0.1,
            checkpoint_filename(16).removesuffix(".zip"): 0.2,
            checkpoint_filename(99).removesuffix(".zip"): 9.9,
            "final_model": 0.3,
        })

        result = run_checkpoint_selection(
            env=None,  # the stub does not use env
            final_model_path=final,
            final_model_timestep=16,
            checkpoints_dir=ckpt_dir,
            selection_dir=sel_dir,
            best_model_path=best,
            n_eval_episodes=1,
            selection_metric="mean_final_pc",
            selection_mode="max",
            checkpoint_paths=current_paths,
        )

        candidate_ids = {c["candidate_id"] for c in result["candidates"]}
        stale_id = checkpoint_filename(99).removesuffix(".zip")
        assert stale_id not in candidate_ids, (
            f"Stale checkpoint leaked into selection: {candidate_ids}"
        )
        # The selected candidate must come from the current run, not the
        # stale file (which would otherwise win on metric value).
        assert result["selected"]["candidate_id"] != stale_id

        # The on-disk JSON also excludes the stale file.
        sel_json = json.loads((sel_dir / "model_selection.json").read_text())
        assert stale_id not in sel_json["all_candidate_ids"]

    def test_default_path_still_uses_discover_checkpoints(self, tmp_path, monkeypatch):
        """Without ``checkpoint_paths``, the legacy discover-from-dir
        behavior is preserved (this is the offline / ad-hoc selection
        path that the closure pass intentionally keeps)."""
        from habconn.training.checkpointing import (
            checkpoint_filename, run_checkpoint_selection,
        )

        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        sel_dir = run_dir / "selection"
        models_dir = run_dir / "models"
        for d in (ckpt_dir, sel_dir, models_dir):
            d.mkdir(parents=True)

        for step in (8, 16):
            (ckpt_dir / checkpoint_filename(step)).write_bytes(b"")
        final = models_dir / "final_model.zip"
        final.write_bytes(b"")

        self._patch_eval(monkeypatch, scores={
            checkpoint_filename(8).removesuffix(".zip"): 0.1,
            checkpoint_filename(16).removesuffix(".zip"): 0.5,
            "final_model": 0.3,
        })

        result = run_checkpoint_selection(
            env=None,
            final_model_path=final,
            final_model_timestep=16,
            checkpoints_dir=ckpt_dir,
            selection_dir=sel_dir,
            best_model_path=models_dir / "best_model.zip",
            n_eval_episodes=1,
            selection_metric="mean_final_pc",
            selection_mode="max",
            # checkpoint_paths intentionally omitted → fallback to
            # discover_checkpoints over the directory.
        )
        ids = [c["candidate_id"] for c in result["candidates"]]
        # Both current-run checkpoints + final, no stale file present.
        assert len(ids) == 3
        assert result["selected"]["candidate_id"] == checkpoint_filename(16).removesuffix(".zip")
