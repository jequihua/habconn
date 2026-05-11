"""Periodic checkpointing and explicit best-model selection.

This module covers Stage 4 milestone 4 of the single-landscape DRL
workflow: every run produces periodic model checkpoints, evaluates
each candidate (checkpoints + the final model) on
``small_vector_001``, and selects an explicit best candidate by a
declared metric.

Surfaces:

- ``CheckpointCallback``: SB3 ``BaseCallback`` that periodically
  saves the current model into ``checkpoints_dir/`` with the stable
  filename pattern ``checkpoint_{num_timesteps:06d}_steps.zip``.
- ``validate_checkpoint_freq``, ``validate_selection_metric``,
  ``validate_selection_mode``: small validators reused by
  ``BaselineConfig`` / ``ExperimentConfig``.
- ``discover_checkpoints``: returns sorted checkpoint paths under a
  directory (parsed from filename, robust to non-checkpoint files).
- ``select_best_candidate``: deterministic selection over a list of
  candidate dicts.
- ``run_checkpoint_selection``: orchestrator that evaluates every
  candidate, picks the best, copies the selected model to
  ``best_model.zip``, and writes
  ``selection/checkpoint_evaluations.json`` and
  ``selection/model_selection.json``.

Semantics notes:

- ``checkpoint_freq`` counts SB3 callback invocations. Under
  ``DummyVecEnv`` with ``n_envs > 1`` each invocation advances
  ``num_timesteps`` by ``n_envs``, so the saved-step values may not
  be exact multiples of ``checkpoint_freq``. The saved filenames
  always reflect the actual ``num_timesteps`` at save time.
- The final-model evaluation that lands in ``baseline_summary.json``
  is **not** reused by the selection step. ``run_checkpoint_selection``
  re-evaluates the final model alongside every checkpoint so all
  candidates share one symmetric evaluation surface. The
  ``evaluation/comparison.{json,csv}`` artifacts continue to use the
  original final-policy ``EvalSummary`` and are not duplicated here.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from habconn.training.evaluation import EvalSummary, evaluate_policy


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


VALID_SELECTION_METRICS: tuple[str, ...] = (
    "mean_final_pc",
    "mean_delta_pc",
    "mean_return",
)
VALID_SELECTION_MODES: tuple[str, ...] = ("max",)


def validate_checkpoint_freq(value: object) -> int:
    """Return ``value`` if it is a positive int. Reject bools and floats."""
    if isinstance(value, bool):
        raise TypeError("checkpoint_freq must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"checkpoint_freq must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"checkpoint_freq must be >= 1, got {value}")
    return value


def validate_selection_metric(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"selection_metric must be str, got {type(value).__name__}"
        )
    if value not in VALID_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {VALID_SELECTION_METRICS}, "
            f"got {value!r}"
        )
    return value


def validate_selection_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"selection_mode must be str, got {type(value).__name__}"
        )
    if value not in VALID_SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {VALID_SELECTION_MODES}, "
            f"got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Filename layout + checkpoint discovery
# ---------------------------------------------------------------------------


CHECKPOINT_FILENAME_PATTERN = "checkpoint_{step:06d}_steps.zip"
_CHECKPOINT_FILENAME_RE = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


def checkpoint_filename(step: int) -> str:
    """Return the stable, sortable filename for a checkpoint at ``step``.

    Six-digit zero-padding keeps lexical and numeric ordering aligned for
    runs up to 999_999 timesteps; ``discover_checkpoints`` parses the
    integer back from the filename so longer training runs still sort
    correctly.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return CHECKPOINT_FILENAME_PATTERN.format(step=step)


def step_from_checkpoint_path(path: Path) -> int:
    """Parse the step count from a checkpoint filename."""
    match = _CHECKPOINT_FILENAME_RE.match(Path(path).name)
    if match is None:
        raise ValueError(
            f"path does not match the checkpoint filename pattern: {path}"
        )
    return int(match.group(1))


def discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Return sorted checkpoint paths under ``checkpoints_dir``.

    Files that do not match the checkpoint filename pattern are ignored
    so that a stray ``.tmp`` / ``.log`` does not break selection. The
    returned list is sorted by parsed step, ascending.
    """
    p = Path(checkpoints_dir)
    if not p.is_dir():
        return []
    matches: list[Path] = []
    for entry in p.iterdir():
        if _CHECKPOINT_FILENAME_RE.match(entry.name):
            matches.append(entry)
    matches.sort(key=step_from_checkpoint_path)
    return matches


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class CheckpointCallback(BaseCallback):
    """Save the current model into ``save_dir`` every ``save_freq`` callback steps.

    The filename is :func:`checkpoint_filename` evaluated at the
    current ``self.num_timesteps`` so it is sortable. ``self.n_calls``
    increments by 1 per callback invocation; under a vectorized env
    each invocation advances ``num_timesteps`` by ``n_envs``, so the
    actual step count between successive saves can be ``save_freq * n_envs``
    rather than exactly ``save_freq``.
    """

    def __init__(self, save_dir: Path, save_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = validate_checkpoint_freq(save_freq)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._saved_paths: list[Path] = []
        self._saved_steps: list[int] = []

    def _on_step(self) -> bool:
        if self.n_calls > 0 and (self.n_calls % self.save_freq == 0):
            step = int(self.num_timesteps)
            target = self.save_dir / checkpoint_filename(step)
            # SB3 ``model.save`` appends ``.zip``; pass the stem so the
            # final filename matches our pattern regardless.
            self.model.save(str(target.with_suffix("")))
            self._saved_paths.append(target)
            self._saved_steps.append(step)
        return True

    @property
    def saved_paths(self) -> list[Path]:
        return list(self._saved_paths)

    @property
    def saved_steps(self) -> list[int]:
        return list(self._saved_steps)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _candidate_id_from_path(model_path: Path, candidate_type: str) -> str:
    """``foo.zip`` → ``foo``; falls back to a type-tagged stem."""
    stem = Path(model_path).stem
    return stem or candidate_type


def select_best_candidate(
    candidates: list[dict[str, Any]],
    metric: str,
    mode: str,
) -> dict[str, Any]:
    """Pick the best candidate by ``metric`` under ``mode``.

    Tie-breaking (deterministic):

    1. higher metric value wins,
    2. among ties, the candidate with the larger ``timestep`` wins,
    3. among further ties, ``candidate_type == "final"`` wins over
       ``"checkpoint"``.

    Each candidate dict must contain ``timestep`` (int),
    ``candidate_type`` (``"checkpoint"`` or ``"final"``), and
    ``evaluation[metric]`` (float).
    """
    metric = validate_selection_metric(metric)
    mode = validate_selection_mode(mode)
    if not candidates:
        raise ValueError("Cannot select a best candidate from an empty list")

    def sort_key(c: dict[str, Any]) -> tuple[float, int, int]:
        timestep = int(c.get("timestep") or 0)
        type_rank = 1 if c.get("candidate_type") == "final" else 0
        return (float(c["evaluation"][metric]), timestep, type_rank)

    # mode == "max" is currently the only supported mode. ``max`` with a
    # tuple key respects the documented tie-breaking order.
    return max(candidates, key=sort_key)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _eval_summary_to_metrics(s: EvalSummary) -> dict[str, float]:
    return {
        "n_episodes": s.n_episodes,
        "mean_return": s.mean_return,
        "mean_final_pc": s.mean_final_pc,
        "mean_delta_pc": s.mean_delta_pc,
        "mean_steps": s.mean_steps,
    }


def _evaluate_model_at_path(
    model_path: Path, env, *, n_episodes: int
) -> EvalSummary:
    """Load a saved SB3 model and evaluate it deterministically."""
    # Imported lazily so the import-graph for unit tests that do not
    # need a model load stays cheap.
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path))
    return evaluate_policy(model, env, n_episodes=n_episodes, deterministic=True)


TIE_BREAK_RULE = (
    "Among candidates of equal metric value, prefer the larger timestep; "
    "if timesteps also tie, prefer 'final' over 'checkpoint'."
)


def run_checkpoint_selection(
    *,
    env,
    final_model_path: Path,
    final_model_timestep: int,
    checkpoints_dir: Path,
    selection_dir: Path,
    best_model_path: Path,
    n_eval_episodes: int,
    selection_metric: str,
    selection_mode: str,
    checkpoint_paths: Optional[Sequence[Path]] = None,
) -> dict[str, Any]:
    """Evaluate all candidates, select best, copy to ``best_model.zip``,
    and write the selection artifacts.

    Parameters
    ----------
    checkpoint_paths
        Optional explicit list of checkpoint paths to consider. When
        provided (typically the trainer hands over
        ``CheckpointCallback.saved_paths``), these are the only
        checkpoint candidates evaluated, sorted by their parsed step,
        and ``discover_checkpoints`` is **not** consulted. This guards
        against stale matching files left in ``checkpoints_dir`` from a
        previous run with the same ``run_name``. When ``None``, the
        function falls back to ``discover_checkpoints(checkpoints_dir)``
        for offline / ad-hoc selection.

    Returns a dict with:
      - ``candidates`` (list of per-candidate records, with ``selected`` flags),
      - ``selected`` (the selected candidate record),
      - ``best_model_path``,
      - ``checkpoint_evaluations_path``,
      - ``model_selection_path``,
      - ``selection_metric`` / ``selection_mode``.
    """
    metric = validate_selection_metric(selection_metric)
    mode = validate_selection_mode(selection_mode)
    if n_eval_episodes < 1:
        raise ValueError(f"n_eval_episodes must be >= 1, got {n_eval_episodes}")

    final_model_path = Path(final_model_path)
    checkpoints_dir = Path(checkpoints_dir)
    selection_dir = Path(selection_dir)
    best_model_path = Path(best_model_path)

    selection_dir.mkdir(parents=True, exist_ok=True)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_paths is None:
        cp_paths = discover_checkpoints(checkpoints_dir)
    else:
        # Accept any iterable of paths; sort by parsed step so the
        # candidate order is stable regardless of caller order.
        cp_paths = sorted(
            (Path(p) for p in checkpoint_paths),
            key=step_from_checkpoint_path,
        )

    candidates: list[dict[str, Any]] = []
    for cp_path in cp_paths:
        candidates.append({
            "candidate_id": _candidate_id_from_path(cp_path, "checkpoint"),
            "model_path": str(cp_path),
            "candidate_type": "checkpoint",
            "timestep": step_from_checkpoint_path(cp_path),
        })

    candidates.append({
        "candidate_id": _candidate_id_from_path(final_model_path, "final"),
        "model_path": str(final_model_path),
        "candidate_type": "final",
        "timestep": int(final_model_timestep),
    })

    # Evaluate every candidate. The trained-policy summary that already
    # lives in baseline_summary.json is not reused here: this path
    # re-evaluates the final model so all candidates share one symmetric
    # evaluation surface. The evaluation/comparison.{json,csv} artifacts
    # are independent and not duplicated.
    for c in candidates:
        s = _evaluate_model_at_path(
            Path(c["model_path"]), env, n_episodes=n_eval_episodes
        )
        c["evaluation"] = _eval_summary_to_metrics(s)

    selected = select_best_candidate(candidates, metric, mode)
    selected_id = selected["candidate_id"]
    for c in candidates:
        c["selected"] = c["candidate_id"] == selected_id

    # Copy the selected model to the canonical best_model.zip path. We
    # use ``shutil.copyfile`` (not ``shutil.copy``) so the destination
    # is a plain file regardless of the source's metadata.
    shutil.copyfile(selected["model_path"], best_model_path)

    checkpoint_evaluations_path = selection_dir / "checkpoint_evaluations.json"
    model_selection_path = selection_dir / "model_selection.json"

    with checkpoint_evaluations_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": metric,
                "selection_mode": mode,
                "candidates": candidates,
            },
            f,
            indent=2,
            default=str,
        )

    selection_payload = {
        "selection_metric": metric,
        "selection_mode": mode,
        "tie_break_rule": TIE_BREAK_RULE,
        "selected_candidate_id": selected_id,
        "selected_candidate_type": selected["candidate_type"],
        "selected_candidate_timestep": selected["timestep"],
        "selected_model_path": selected["model_path"],
        "best_model_path": str(best_model_path),
        "selected_evaluation": selected["evaluation"],
        "all_candidate_ids": [c["candidate_id"] for c in candidates],
    }
    with model_selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2, default=str)

    return {
        "candidates": candidates,
        "selected": selected,
        "best_model_path": str(best_model_path),
        "checkpoint_evaluations_path": str(checkpoint_evaluations_path),
        "model_selection_path": str(model_selection_path),
        "selection_metric": metric,
        "selection_mode": mode,
    }
