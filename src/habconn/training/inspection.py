"""Feature inspection and deployment action-trace artifacts.

Stage 4 milestone 6: make the v2 observation contract and the
deployed policy's per-step decisions reviewable from JSON/CSV.

Surfaces:

- ``observation_schema(obs, *, extractor_consumed_keys)``: schema of
  the 14-key v2 observation with consumed-vs-unused flags relative
  to the current ``FlatObsExtractor``.
- ``summarize_observation_features(obs)``: compact per-key numeric
  / boolean stats over a single observation dict.
- ``build_action_trace_rows(trace_steps)``: expand a per-step trace
  into one row per (step, candidate slot) for the CSV artifact.
- ``write_inspection_artifacts(output_dir, *, schema,
  feature_summary, trace_steps, run_name)``: write the four
  inspection files under ``output_dir``.

Scope notes:
- This is **inspection**, not feature attribution. No SHAP,
  no permutation importance, no per-feature gradient analysis.
- The trace records what the policy *saw* and *chose*, not why.
- All artifacts are JSON-serializable without ``default=str``
  fallback for normal numpy dtypes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


OBSERVATION_VERSION = "v2"
N_OBSERVATION_KEYS = 14

# v2 key groups.
ACTION_KEYS = ("action_mask", "candidate_ids", "candidate_costs", "candidate_areas")
NODE_KEYS = (
    "selected_mask", "node_mask", "node_costs", "node_areas", "eligibility_mask",
)
GLOBAL_KEYS = (
    "remaining_budget", "budget_fraction", "step_count",
    "selected_fraction", "current_pc",
)

# Keys actually consumed by FlatObsExtractor (see
# models/extractors/padded_mlp.py). This must stay in sync with the
# extractor; the unit test fixes it as a regression guard.
FLAT_EXTRACTOR_CONSUMED_KEYS: tuple[str, ...] = (
    "action_mask",
    "candidate_costs",
    "candidate_areas",
    "remaining_budget",
    "budget_fraction",
    "step_count",
    "selected_fraction",
    "current_pc",
)

# Output filenames.
OBSERVATION_SCHEMA_FILENAME = "observation_schema.json"
FEATURE_SUMMARY_FILENAME = "feature_summary.json"
ACTION_TRACE_JSON_FILENAME = "deployment_action_trace.json"
ACTION_TRACE_CSV_FILENAME = "deployment_action_trace.csv"

_TRACE_CSV_COLUMNS: tuple[str, ...] = (
    "step",
    "slot",
    "pu_id",
    "valid",
    "chosen",
    "candidate_cost",
    "candidate_area",
    "remaining_budget_before",
    "current_pc_before",
    "reward_after",
    "pc_after",
)

_GROUP_ORDER = {"action": 0, "node": 1, "global": 2, "unknown": 3}


# ---------------------------------------------------------------------------
# Observation schema
# ---------------------------------------------------------------------------


def _group_for(key: str) -> str:
    if key in ACTION_KEYS:
        return "action"
    if key in NODE_KEYS:
        return "node"
    if key in GLOBAL_KEYS:
        return "global"
    return "unknown"


_NOTES = {
    "action_mask": "True for valid candidate slots; padded slots are False.",
    "candidate_ids": "Planning-unit ID per candidate slot; -1 for padded slot.",
    "candidate_costs": "Cost per candidate slot; 0.0 for padded slot.",
    "candidate_areas": "Area per candidate slot; 0.0 for padded slot.",
    "selected_mask": "True for already-selected planning units.",
    "node_mask": "True for real planning units; False for padding to n_max.",
    "node_costs": "Cost per node; 0.0 for padded node.",
    "node_areas": "Area per node; 0.0 for padded node.",
    "eligibility_mask": "True for nodes the policy may still pick.",
    "remaining_budget": "Budget left before this action.",
    "budget_fraction": "remaining / initial budget.",
    "step_count": "Cumulative steps taken in the episode.",
    "selected_fraction": "n_selected / n_planning_units.",
    "current_pc": "Most recent Probability of Connectivity value.",
}


def observation_schema(
    observation: dict[str, np.ndarray],
    *,
    extractor_consumed_keys: Iterable[str] = FLAT_EXTRACTOR_CONSUMED_KEYS,
) -> dict[str, Any]:
    """Return a JSON-serializable schema describing the v2 observation.

    Each key record carries ``group``, ``shape``, ``dtype``,
    ``consumed_by_flat_extractor``, and ``notes``. A warning is added
    when node-level arrays are present but the extractor does not
    consume them, which is the current state of the package.
    """
    consumed = set(extractor_consumed_keys)
    keys: list[dict[str, Any]] = []
    for name, arr in observation.items():
        a = np.asarray(arr)
        keys.append({
            "name": name,
            "group": _group_for(name),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "consumed_by_flat_extractor": name in consumed,
            "notes": _NOTES.get(name),
        })
    keys.sort(key=lambda r: (_GROUP_ORDER[r["group"]], r["name"]))

    unused = [k["name"] for k in keys if not k["consumed_by_flat_extractor"]]
    warnings: list[str] = []
    if any(k["group"] == "node" and not k["consumed_by_flat_extractor"] for k in keys):
        warnings.append(
            "Node-level arrays are present in the v2 observation but are not "
            "consumed by the current FlatObsExtractor; they are reserved for "
            "a future set/graph encoder."
        )

    return {
        "observation_version": OBSERVATION_VERSION,
        "n_keys": len(keys),
        "keys": keys,
        "flat_extractor_consumed_keys": sorted(consumed),
        "unused_by_flat_extractor": unused,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Feature summary
# ---------------------------------------------------------------------------


def _per_key_stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr)
    stats: dict[str, Any] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }
    if a.dtype == np.bool_:
        true_count = int(a.sum())
        false_count = int(a.size - true_count)
        stats["true_count"] = true_count
        stats["false_count"] = false_count
        return stats

    is_float = np.issubdtype(a.dtype, np.floating)
    if is_float:
        finite_mask = np.isfinite(a)
        nan_count = int(np.isnan(a).sum())
    else:
        finite_mask = np.ones_like(a, dtype=bool)
        nan_count = 0
    finite_count = int(finite_mask.sum())
    stats["finite_count"] = finite_count
    stats["nan_count"] = nan_count
    if finite_count > 0:
        finite_vals = a[finite_mask]
        stats["min"] = float(np.min(finite_vals))
        stats["max"] = float(np.max(finite_vals))
        stats["mean"] = float(np.mean(finite_vals))
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
    return stats


def summarize_observation_features(
    observation: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compact per-key stats over one observation. JSON-serializable."""
    return {name: _per_key_stats(np.asarray(arr)) for name, arr in observation.items()}


# ---------------------------------------------------------------------------
# Action trace
# ---------------------------------------------------------------------------


def _row_for_slot(step: dict[str, Any], slot: int) -> dict[str, Any]:
    mask = step["action_mask"]
    cand_ids = step["candidate_ids"]
    cand_costs = step["candidate_costs"]
    cand_areas = step["candidate_areas"]
    k = len(mask)
    valid = bool(mask[slot]) if slot < k else False
    return {
        "step": int(step["step"]),
        "slot": int(slot),
        "pu_id": int(cand_ids[slot]) if slot < len(cand_ids) else -1,
        "valid": valid,
        "chosen": (slot == int(step["chosen_slot"])),
        "candidate_cost": float(cand_costs[slot]) if slot < len(cand_costs) else 0.0,
        "candidate_area": float(cand_areas[slot]) if slot < len(cand_areas) else 0.0,
        "remaining_budget_before": float(step["remaining_budget_before"]),
        "current_pc_before": float(step["current_pc_before"]),
        "reward_after": float(step["reward_after"]),
        "pc_after": float(step["pc_after"]),
    }


def build_action_trace_rows(
    trace_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand per-step trace dicts into one row per (step, slot)."""
    rows: list[dict[str, Any]] = []
    for step in trace_steps:
        k = len(step["action_mask"])
        for slot in range(k):
            rows.append(_row_for_slot(step, slot))
    return rows


def _trace_steps_for_json(
    trace_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in trace_steps:
        k = len(step["action_mask"])
        slots = []
        for slot in range(k):
            row = _row_for_slot(step, slot)
            slots.append({
                "slot": row["slot"],
                "pu_id": row["pu_id"],
                "valid": row["valid"],
                "chosen": row["chosen"],
                "candidate_cost": row["candidate_cost"],
                "candidate_area": row["candidate_area"],
            })
        out.append({
            "step": int(step["step"]),
            "chosen_slot": int(step["chosen_slot"]),
            "chosen_pu_id": int(step["chosen_pu_id"]),
            "valid_action_count": int(step["valid_action_count"]),
            "remaining_budget_before": float(step["remaining_budget_before"]),
            "current_pc_before": float(step["current_pc_before"]),
            "reward_after": float(step["reward_after"]),
            "pc_after": float(step["pc_after"]),
            "selected_pu_ids_after": [int(p) for p in step["selected_pu_ids_after"]],
            "candidate_slots": slots,
        })
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_inspection_artifacts(
    output_dir: Path,
    *,
    schema: dict[str, Any],
    feature_summary: dict[str, Any],
    trace_steps: list[dict[str, Any]],
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    """Write the four inspection artifacts under ``output_dir``.

    Returns a dict of paths and small counts useful to the trainer's
    run summary. The CSV is always written with a header row even
    when ``trace_steps`` is empty.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_dir / OBSERVATION_SCHEMA_FILENAME
    feature_path = output_dir / FEATURE_SUMMARY_FILENAME
    trace_json_path = output_dir / ACTION_TRACE_JSON_FILENAME
    trace_csv_path = output_dir / ACTION_TRACE_CSV_FILENAME

    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    with feature_path.open("w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2)

    rows = build_action_trace_rows(trace_steps)
    payload = {
        "run_name": run_name,
        "n_steps": len(trace_steps),
        "n_rows": len(rows),
        "steps": _trace_steps_for_json(trace_steps),
    }
    with trace_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with trace_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_TRACE_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "observation_schema_path": str(schema_path),
        "feature_summary_path": str(feature_path),
        "deployment_action_trace_json_path": str(trace_json_path),
        "deployment_action_trace_csv_path": str(trace_csv_path),
        "n_deployment_trace_steps": len(trace_steps),
        "n_deployment_trace_rows": len(rows),
    }
