"""Experiment contract for single-landscape training runs.

This module defines the run configuration, the on-disk output layout,
and the metadata captured for every training run. It is the single
entry point for routing the existing baseline through a stable,
reviewable artifact contract.

Output layout for a run::

    output_root/run_name/
        config.json              -- resolved ExperimentConfig
        metadata.json            -- environment/version/git/timestamp metadata
        history.jsonl            -- per-episode training history (callback)
        baseline_summary.json    -- training + trained-policy evaluation
        evaluation/
            comparison.json      -- trained policy vs simple baselines
            comparison.csv       -- compact method-level table
        checkpoints/
            checkpoint_NNNNNN_steps.zip   -- periodic SB3 checkpoints
        selection/
            checkpoint_evaluations.json   -- per-candidate evaluation
            model_selection.json          -- selected candidate + rule
        deployment/
            deployment_summary.json       -- deployed model + selected PUs
            selected_planning_units.gpkg  -- selected geometries (GeoPackage)
            selected_planning_units.csv   -- review-friendly table
        inspection/
            observation_schema.json       -- v2 obs keys + consumed flags
            feature_summary.json          -- per-key stats over the initial obs
            deployment_action_trace.json  -- per-step candidate + choice trace
            deployment_action_trace.csv   -- one row per (step, slot)
        models/
            final_model.zip      -- final SB3 model
            best_model.zip       -- copy of the selected candidate

Scope notes:
- Vectorized env training is wired through ``n_envs`` and
  ``training/vecenv.py``.
- Checkpointing + best-model selection is wired through
  ``checkpoint_freq`` / ``selection_metric`` / ``selection_mode``
  and ``training/checkpointing.py``.
- Deployment export is wired through ``training/deployment.py``.
- Feature inspection + deployment action trace artifacts are wired
  through ``training/inspection.py``.
- This contract still does not implement transfer-learning data,
  multi-landscape splits, graph/set encoders, or richer reporting;
  those will hang off this contract in later stages.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from habconn.training.trainer import BaselineConfig, train_baseline


# A run_name must be a slug-like single path segment so that
# ``output_root / run_name`` can never escape ``output_root``. Allowed
# characters: ASCII letters, digits, underscore, hyphen, and dot. The
# first character must be a letter, digit, or underscore so the name
# cannot start with a dot (avoiding ``.``, ``..``, and dotfile-style
# hidden directories) or a hyphen (which is harder to handle as a CLI
# argument later).
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def _validate_run_name(name: object) -> str:
    """Validate a run_name and return it.

    Raises ``TypeError`` for non-string input and ``ValueError`` for any
    name that could break the contract's per-run isolation guarantee.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"run_name must be a string, got {type(name).__name__}"
        )
    if not name or not name.strip():
        raise ValueError("run_name must be a non-empty, non-whitespace string")
    if name in (".", ".."):
        raise ValueError(f"run_name {name!r} is reserved")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"run_name must not contain path separators: {name!r}"
        )
    if ".." in name:
        raise ValueError(
            f"run_name must not contain '..' (path traversal): {name!r}"
        )
    # Reject absolute or drive-qualified paths (e.g. 'C:foo' on Windows).
    if Path(name).is_absolute() or (len(name) >= 2 and name[1] == ":"):
        raise ValueError(f"run_name must not be an absolute path: {name!r}")
    if not _RUN_NAME_RE.match(name):
        raise ValueError(
            f"run_name {name!r} must match [A-Za-z0-9_][A-Za-z0-9_.-]* "
            f"(letters, digits, underscore, hyphen, dot; not starting "
            f"with '.' or '-')"
        )
    return name


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Resolved configuration for a single-landscape training run.

    All paths are explicit. ``output_root`` together with ``run_name``
    determines the run directory; nothing else writes outside it.
    """

    # Identity
    run_name: str = "baseline"
    seed: int = 42

    # Data + backend paths
    data_dir: Path = Path("data/examples/small_vector_001")
    graphab_jar: Path = Path("tools/graphab.jar")
    work_root: Path = Path("tmp/training_runs")

    # Output root (run directory is output_root / run_name)
    output_root: Path = Path("tmp/experiments")

    # Environment knobs
    budget: int = 3
    k: int = 10

    # Training knobs (PPO)
    total_timesteps: int = 50
    learning_rate: float = 3e-4
    n_steps: int = 8
    batch_size: int = 4
    n_epochs: int = 2
    gamma: float = 0.99

    # Evaluation
    n_eval_episodes: int = 1

    # Vectorized training. ``n_envs == 1`` preserves the legacy single-env
    # path. ``n_envs > 1`` builds a DummyVecEnv with isolated Graphab work
    # roots and deterministic per-worker seeds.
    n_envs: int = 1

    # Checkpointing + best-model selection.
    checkpoint_freq: int = 16
    selection_metric: str = "mean_final_pc"
    selection_mode: str = "max"

    # TensorBoard logging. Off by default. When ``enable_tensorboard``
    # is True and ``tensorboard_log`` is None, the default log directory
    # is ``ExperimentPaths.tensorboard_dir`` (``output_root/run_name/
    # tensorboard/``). Setting ``tensorboard_log`` explicitly overrides
    # the default. ``tb_log_name`` is the SB3 sub-run-name appended
    # under the log directory.
    enable_tensorboard: bool = False
    tensorboard_log: Optional[Path] = None
    tb_log_name: str = "ppo"

    def __post_init__(self) -> None:
        # Coerce string paths to Path so external (JSON) callers work.
        self.data_dir = Path(self.data_dir)
        self.graphab_jar = Path(self.graphab_jar)
        self.work_root = Path(self.work_root)
        self.output_root = Path(self.output_root)
        # Strict validation so the contract's isolation promise holds:
        # ``output_root / run_name`` must always stay inside ``output_root``.
        self.run_name = _validate_run_name(self.run_name)
        if not isinstance(self.n_envs, int) or isinstance(self.n_envs, bool):
            raise TypeError(
                f"n_envs must be int, got {type(self.n_envs).__name__}"
            )
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {self.n_envs}")
        # Validate checkpointing + selection knobs in-place.
        from habconn.training.checkpointing import (
            validate_checkpoint_freq,
            validate_selection_metric,
            validate_selection_mode,
        )
        self.checkpoint_freq = validate_checkpoint_freq(self.checkpoint_freq)
        self.selection_metric = validate_selection_metric(self.selection_metric)
        self.selection_mode = validate_selection_mode(self.selection_mode)
        # TensorBoard knobs.
        if not isinstance(self.enable_tensorboard, bool):
            raise TypeError(
                f"enable_tensorboard must be bool, got "
                f"{type(self.enable_tensorboard).__name__}"
            )
        if self.tensorboard_log is not None:
            self.tensorboard_log = Path(self.tensorboard_log)
        if not isinstance(self.tb_log_name, str) or not self.tb_log_name.strip():
            raise ValueError("tb_log_name must be a non-empty string")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved on-disk layout for a single experiment run."""

    run_dir: Path
    config_path: Path
    metadata_path: Path
    history_path: Path
    summary_path: Path
    models_dir: Path
    model_path: Path
    best_model_path: Path
    evaluation_dir: Path
    comparison_json_path: Path
    comparison_csv_path: Path
    checkpoints_dir: Path
    selection_dir: Path
    checkpoint_evaluations_path: Path
    model_selection_path: Path
    deployment_dir: Path
    deployment_summary_path: Path
    selected_planning_units_gpkg_path: Path
    selected_planning_units_csv_path: Path
    inspection_dir: Path
    observation_schema_path: Path
    feature_summary_path: Path
    deployment_action_trace_json_path: Path
    deployment_action_trace_csv_path: Path
    tensorboard_dir: Path

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "ExperimentPaths":
        run_dir = config.output_root / config.run_name
        # Defense in depth: even after run_name validation in
        # ExperimentConfig, verify the resolved run directory is contained
        # under the resolved output root. This catches any future bypass
        # where a caller bypasses ExperimentConfig.__post_init__.
        try:
            resolved_root = config.output_root.resolve()
            resolved_run = run_dir.resolve()
            resolved_run.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"resolved run directory {resolved_run} is not contained "
                f"under output_root {resolved_root}"
            ) from exc
        models_dir = run_dir / "models"
        evaluation_dir = run_dir / "evaluation"
        checkpoints_dir = run_dir / "checkpoints"
        selection_dir = run_dir / "selection"
        deployment_dir = run_dir / "deployment"
        inspection_dir = run_dir / "inspection"
        tensorboard_dir = run_dir / "tensorboard"
        return cls(
            run_dir=run_dir,
            config_path=run_dir / "config.json",
            metadata_path=run_dir / "metadata.json",
            history_path=run_dir / "history.jsonl",
            summary_path=run_dir / "baseline_summary.json",
            models_dir=models_dir,
            model_path=models_dir / "final_model.zip",
            best_model_path=models_dir / "best_model.zip",
            evaluation_dir=evaluation_dir,
            comparison_json_path=evaluation_dir / "comparison.json",
            comparison_csv_path=evaluation_dir / "comparison.csv",
            checkpoints_dir=checkpoints_dir,
            selection_dir=selection_dir,
            checkpoint_evaluations_path=selection_dir / "checkpoint_evaluations.json",
            model_selection_path=selection_dir / "model_selection.json",
            deployment_dir=deployment_dir,
            deployment_summary_path=deployment_dir / "deployment_summary.json",
            selected_planning_units_gpkg_path=deployment_dir / "selected_planning_units.gpkg",
            selected_planning_units_csv_path=deployment_dir / "selected_planning_units.csv",
            inspection_dir=inspection_dir,
            observation_schema_path=inspection_dir / "observation_schema.json",
            feature_summary_path=inspection_dir / "feature_summary.json",
            deployment_action_trace_json_path=inspection_dir / "deployment_action_trace.json",
            deployment_action_trace_csv_path=inspection_dir / "deployment_action_trace.csv",
            tensorboard_dir=tensorboard_dir,
        )

    def ensure(self) -> None:
        """Create the run directory and every subdirectory the contract
        advertises (``models/``, ``evaluation/``, ``checkpoints/``,
        ``selection/``, ``deployment/``, ``inspection/``).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.selection_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.inspection_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Serialization + metadata
# ---------------------------------------------------------------------------


def serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-serializable dict of the resolved config."""
    raw = asdict(config)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in raw.items()}


def _safe_dependency_versions() -> dict[str, Optional[str]]:
    """Best-effort version probe for cheap-to-import deps. None on failure."""
    pkgs = (
        "numpy",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "sb3_contrib",
    )
    out: dict[str, Optional[str]] = {}
    for name in pkgs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
        except Exception:
            out[name] = None
    return out


def _safe_git_commit(cwd: Path) -> Optional[str]:
    """Return the short git commit hash if available, else None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _safe_package_info() -> dict[str, Optional[str]]:
    """Resolve habconn package version and import path."""
    info: dict[str, Optional[str]] = {"version": None, "path": None}
    try:
        import habconn

        info["version"] = getattr(habconn, "__version__", None)
        info["path"] = getattr(habconn, "__file__", None)
    except Exception:
        pass
    return info


def collect_metadata(config: ExperimentConfig) -> dict[str, Any]:
    """Collect environment metadata for a run.

    Resilient by construction: any single field that cannot be resolved
    becomes ``null`` rather than raising.
    """
    pkg = _safe_package_info()
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_name": config.run_name,
        "seed": config.seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "habconn_version": pkg["version"],
        "habconn_path": pkg["path"],
        "git_commit": _safe_git_commit(Path.cwd()),
        "dependencies": _safe_dependency_versions(),
        "paths": {
            "data_dir": str(config.data_dir),
            "graphab_jar": str(config.graphab_jar),
            "work_root": str(config.work_root),
            "output_root": str(config.output_root),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def resolve_tensorboard_log(
    config: ExperimentConfig, paths: ExperimentPaths
) -> Optional[Path]:
    """Resolve the SB3 ``tensorboard_log`` value for this run.

    ``None`` when TensorBoard is disabled. When enabled and the user
    did not pass an explicit ``tensorboard_log``, defaults to
    ``paths.tensorboard_dir``.
    """
    if not config.enable_tensorboard:
        return None
    if config.tensorboard_log is not None:
        return Path(config.tensorboard_log)
    return paths.tensorboard_dir


def _baseline_config_from_experiment(
    config: ExperimentConfig, paths: ExperimentPaths
) -> BaselineConfig:
    """Translate an ExperimentConfig + Paths into a BaselineConfig."""
    return BaselineConfig(
        data_dir=config.data_dir,
        graphab_jar=config.graphab_jar,
        work_root=config.work_root,
        budget=config.budget,
        k=config.k,
        seed=config.seed,
        total_timesteps=config.total_timesteps,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        n_eval_episodes=config.n_eval_episodes,
        n_envs=config.n_envs,
        checkpoint_freq=config.checkpoint_freq,
        selection_metric=config.selection_metric,
        selection_mode=config.selection_mode,
        output_dir=paths.run_dir,
        history_path=paths.history_path,
        summary_path=paths.summary_path,
        model_path=paths.model_path,
        checkpoints_dir=paths.checkpoints_dir,
        selection_dir=paths.selection_dir,
        best_model_path=paths.best_model_path,
        tensorboard_log=resolve_tensorboard_log(config, paths),
        tb_log_name=config.tb_log_name,
    )


def setup_experiment(config: ExperimentConfig) -> ExperimentPaths:
    """Resolve paths, create directories, and persist config + metadata.

    This is the part of ``run_experiment`` that does not require Graphab,
    so it can be exercised cheaply in tests and from offline tooling.
    """
    paths = ExperimentPaths.from_config(config)
    paths.ensure()
    _write_json(paths.config_path, serialize_config(config))
    _write_json(paths.metadata_path, collect_metadata(config))
    return paths


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a single-landscape training experiment under the contract.

    Side effects (in ``output_root/run_name``):
      - ``config.json`` (resolved config)
      - ``metadata.json`` (timestamp, versions, git commit, paths)
      - ``history.jsonl`` (per-episode training history)
      - ``baseline_summary.json`` (training + evaluation summary)
      - ``models/final_model.zip`` (final SB3 model)

    Returns the ``baseline_summary.json`` payload augmented with a
    ``run_dir`` field for convenience.
    """
    paths = setup_experiment(config)
    baseline_cfg = _baseline_config_from_experiment(config, paths)
    summary = train_baseline(baseline_cfg)
    summary["run_dir"] = str(paths.run_dir)
    summary["run_name"] = config.run_name
    summary["config_path"] = str(paths.config_path)
    summary["metadata_path"] = str(paths.metadata_path)
    # Comparison paths are also produced by train_baseline; mirror them
    # at the experiment level so callers can locate every artifact from
    # the returned summary alone.
    summary.setdefault("comparison_json_path", str(paths.comparison_json_path))
    summary.setdefault("comparison_csv_path", str(paths.comparison_csv_path))
    summary.setdefault("evaluation_dir", str(paths.evaluation_dir))
    # Checkpointing + selection paths.
    summary.setdefault("checkpoints_dir", str(paths.checkpoints_dir))
    summary.setdefault("selection_dir", str(paths.selection_dir))
    summary.setdefault("best_model_path", str(paths.best_model_path))
    summary.setdefault(
        "checkpoint_evaluations_path", str(paths.checkpoint_evaluations_path)
    )
    summary.setdefault("model_selection_path", str(paths.model_selection_path))
    # Deployment paths.
    summary.setdefault("deployment_dir", str(paths.deployment_dir))
    summary.setdefault(
        "deployment_summary_path", str(paths.deployment_summary_path)
    )
    summary.setdefault(
        "selected_planning_units_gpkg_path",
        str(paths.selected_planning_units_gpkg_path),
    )
    summary.setdefault(
        "selected_planning_units_csv_path",
        str(paths.selected_planning_units_csv_path),
    )
    # Inspection paths.
    summary.setdefault("inspection_dir", str(paths.inspection_dir))
    summary.setdefault(
        "observation_schema_path", str(paths.observation_schema_path)
    )
    summary.setdefault(
        "feature_summary_path", str(paths.feature_summary_path)
    )
    summary.setdefault(
        "deployment_action_trace_json_path",
        str(paths.deployment_action_trace_json_path),
    )
    summary.setdefault(
        "deployment_action_trace_csv_path",
        str(paths.deployment_action_trace_csv_path),
    )
    # TensorBoard surfacing. ``tensorboard_dir`` always points at the
    # contract default location; ``tensorboard_log`` reflects what was
    # actually passed to MaskablePPO (``None`` when disabled).
    summary.setdefault("tensorboard_dir", str(paths.tensorboard_dir))
    resolved_tb = resolve_tensorboard_log(config, paths)
    summary.setdefault("enable_tensorboard", bool(config.enable_tensorboard))
    summary.setdefault(
        "tensorboard_log", str(resolved_tb) if resolved_tb is not None else None,
    )
    summary.setdefault("tb_log_name", config.tb_log_name)
    return summary
