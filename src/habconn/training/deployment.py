"""Deployment of the selected best model on a single landscape.

Stage 4 milestone 5: load ``models/best_model.zip``, run one
deterministic masked deployment episode against the dedicated
single-env evaluation env, and export:

- ``deployment/deployment_summary.json`` — model + selected PUs +
  episode metrics + artifact paths
- ``deployment/selected_planning_units.gpkg`` — GIS-friendly
  selected geometries (GeoPackage)
- ``deployment/selected_planning_units.csv`` — review-friendly
  per-row table (no geometry blob)

Selection order is preserved end-to-end: the order of action choices
during the deployment episode is the order of ``selected_pu_ids`` in
the summary and the value of the ``selection_order`` column in the
exported tables.

This is **single-landscape** deployment for ``small_vector_001``. It
is not a transfer-learning result and not a scientific optimality
proof.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


GEOPACKAGE_FILENAME = "selected_planning_units.gpkg"
CSV_FILENAME = "selected_planning_units.csv"
SUMMARY_FILENAME = "deployment_summary.json"


def _scalar(arr) -> float:
    """Extract a Python float from a shape-(1,) array or scalar."""
    return float(np.asarray(arr).reshape(-1)[0])


def _run_deployment_episode(env, model) -> dict[str, Any]:
    """Run one deterministic masked episode and capture per-step traces.

    The selection order is the order of the actions emitted by the
    policy; it is recovered from ``info['selected_pu_ids']`` at the
    end of the episode (the env appends to that list on every valid
    step).

    The returned dict additionally carries ``initial_observation`` and
    ``trace_steps`` so the inspection layer can produce the v2 schema,
    feature summary, and per-step action trace without re-running the
    episode. Those two fields contain numpy arrays and are not
    intended to be serialized directly.
    """
    obs, info = env.reset()
    baseline_pc = float(info.get("pc_value", 0.0))
    initial_observation = {k: np.asarray(v).copy() for k, v in obs.items()}

    step_rewards: list[float] = []
    step_pc_values: list[float] = []
    trace_steps: list[dict[str, Any]] = []
    episode_return = 0.0
    done = False
    step_idx = 0

    while not done:
        masks = env.action_masks()
        # If the env exposes no valid candidate (e.g. budget too small
        # for any planning unit), terminate the deployment episode
        # cleanly rather than invoke the policy on an all-False mask.
        # The exported artifacts still document this run honestly.
        if hasattr(masks, "any") and not masks.any():
            break

        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        # Snapshot the pre-step observation slices the trace needs.
        action_mask_arr = np.asarray(obs["action_mask"], dtype=bool).copy()
        candidate_ids_arr = np.asarray(obs["candidate_ids"]).astype(int, copy=True)
        candidate_costs_arr = np.asarray(obs["candidate_costs"], dtype=np.float32).copy()
        candidate_areas_arr = np.asarray(obs["candidate_areas"], dtype=np.float32).copy()
        remaining_budget_before = _scalar(obs["remaining_budget"])
        current_pc_before = _scalar(obs["current_pc"])
        chosen_pu_id = (
            int(candidate_ids_arr[action]) if action < len(candidate_ids_arr) else -1
        )

        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1
        pc_after = float(info.get("pc_value", 0.0))
        episode_return += float(reward)
        step_rewards.append(float(reward))
        step_pc_values.append(pc_after)

        trace_steps.append({
            "step": step_idx,
            "chosen_slot": action,
            "chosen_pu_id": chosen_pu_id,
            "valid_action_count": int(action_mask_arr.sum()),
            "remaining_budget_before": remaining_budget_before,
            "current_pc_before": current_pc_before,
            "reward_after": float(reward),
            "pc_after": pc_after,
            "selected_pu_ids_after": [int(p) for p in info.get("selected_pu_ids", [])],
            "action_mask": action_mask_arr,
            "candidate_ids": candidate_ids_arr,
            "candidate_costs": candidate_costs_arr,
            "candidate_areas": candidate_areas_arr,
        })

        done = terminated or truncated

    final_pc = step_pc_values[-1] if step_pc_values else baseline_pc
    selected_pu_ids = [int(pid) for pid in info.get("selected_pu_ids", [])]

    return {
        "selected_pu_ids": selected_pu_ids,
        "episode_steps": len(step_rewards),
        "deployment_return": episode_return,
        "baseline_pc": baseline_pc,
        "final_pc": final_pc,
        "delta_pc_total": final_pc - baseline_pc,
        "step_rewards": step_rewards,
        "step_pc_values": step_pc_values,
        # Inspection side channel (numpy; not JSON-serialized):
        "initial_observation": initial_observation,
        "trace_steps": trace_steps,
    }


def _ordered_selection_frame(problem, selected_pu_ids: list[int]):
    """Return a GeoDataFrame of the selected planning units in selection order.

    Adds a ``selection_order`` column (0-based). If no units were
    selected, returns an empty GeoDataFrame that still has the
    ``selection_order`` column so the schema is stable.
    """
    gdf = problem.selected_geodataframe(selected_pu_ids)
    if len(selected_pu_ids) == 0:
        if "selection_order" not in gdf.columns:
            gdf = gdf.copy()
            gdf["selection_order"] = []
        return gdf

    order_map = {int(pid): i for i, pid in enumerate(selected_pu_ids)}
    id_col = problem.internal_id_column
    gdf = gdf.copy()
    gdf["selection_order"] = gdf[id_col].astype(int).map(order_map)
    return gdf.sort_values("selection_order", kind="stable").reset_index(drop=True)


def _csv_columns_for(problem) -> list[str]:
    """Lean review-friendly column set; geometry is intentionally omitted."""
    cols = [problem.internal_id_column]
    if problem.id_column and problem.id_column != problem.internal_id_column:
        cols.append(problem.id_column)
    if problem.cost_column:
        cols.append(problem.cost_column)
    cols.append("selection_order")
    return cols


def _write_geopackage(gdf, path: Path) -> None:
    """Write a GeoPackage. Overwrites any prior file at ``path``.

    For an **empty** selection, ``GeoDataFrame.to_file`` is unreliable
    across fiona/pyogrio versions, so we write an empty placeholder
    file at the contract path. Reader-side code (e.g. the integration
    test) is expected to skip GeoPackage parsing for empty selections.

    For a **non-empty** selection, writer failures are propagated to
    the caller rather than silently masked: a broken write would
    otherwise produce a misleading empty placeholder that the
    integration test (or a human reviewer) would interpret as a real
    "no units selected" result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if len(gdf) == 0:
        path.write_bytes(b"")
        return
    gdf.to_file(path, driver="GPKG")


def _write_csv(gdf, path: Path, columns: list[str]) -> None:
    """Write a lean CSV (no geometry column)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    available = [c for c in columns if c in gdf.columns]
    if len(gdf) == 0:
        path.write_text(",".join(available) + "\n", encoding="utf-8")
        return
    gdf[available].to_csv(path, index=False)


def run_deployment_export(
    *,
    model_path: Path,
    env,
    problem,
    output_dir: Path,
    run_name: Optional[str] = None,
    source_model_selection_path: Optional[Path] = None,
    budget: Optional[int] = None,
    k: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Load ``model_path``, deploy on ``env``, and write deployment artifacts.

    Returns the in-memory deployment summary dict that was also
    serialized to ``deployment_summary.json``.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path))
    episode = _run_deployment_episode(env, model)

    selected_pu_ids: list[int] = episode["selected_pu_ids"]
    gdf = _ordered_selection_frame(problem, selected_pu_ids)

    gpkg_path = output_dir / GEOPACKAGE_FILENAME
    csv_path = output_dir / CSV_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    _write_geopackage(gdf, gpkg_path)
    _write_csv(gdf, csv_path, _csv_columns_for(problem))

    summary: dict[str, Any] = {
        "run_name": run_name,
        "model_path": str(model_path),
        "model_selection_path": (
            str(source_model_selection_path)
            if source_model_selection_path is not None
            else None
        ),
        "deterministic": True,
        "budget": budget,
        "k": k,
        "data_dir": str(data_dir) if data_dir is not None else None,
        "selected_pu_ids": selected_pu_ids,
        "n_selected": len(selected_pu_ids),
        "deployment_return": episode["deployment_return"],
        "baseline_pc": episode["baseline_pc"],
        "final_pc": episode["final_pc"],
        "delta_pc_total": episode["delta_pc_total"],
        "episode_steps": episode["episode_steps"],
        "step_rewards": episode["step_rewards"],
        "step_pc_values": episode["step_pc_values"],
        "selected_planning_units_gpkg_path": str(gpkg_path),
        "selected_planning_units_csv_path": str(csv_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["deployment_summary_path"] = str(summary_path)
    # Inspection side channel for the trainer / inspection writer.
    # Underscore-prefixed: not part of the JSON contract; ignored by
    # downstream consumers that only read the on-disk summary file.
    summary["_initial_observation"] = episode["initial_observation"]
    summary["_trace_steps"] = episode["trace_steps"]
    return summary
