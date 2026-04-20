"""Per-planning-unit (node-level) feature builders.

Node-level features are aligned with the padded planning-unit index:
for each of the first ``n_max`` slots, they report values for the
corresponding planning unit (or 0/False for padding slots).

All builders are deterministic and pure: same (problem, state, n_max)
input produces same output.
"""

from __future__ import annotations

import numpy as np

from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def build_node_costs(
    problem: VectorConnectivityProblem,
    *,
    n_max: int,
) -> np.ndarray:
    """Restoration cost per planning unit, padded to n_max.

    Returns a float32 array of shape (n_max,). Padded slots are 0.0.
    """
    if problem.n_planning_units > n_max:
        raise ValueError(
            f"n_max={n_max} is smaller than n_planning_units="
            f"{problem.n_planning_units}"
        )
    out = np.zeros(n_max, dtype=np.float32)
    costs = problem.planning_units[problem.cost_column].astype(float).to_numpy()
    out[: problem.n_planning_units] = costs
    return out


def build_node_areas(
    problem: VectorConnectivityProblem,
    *,
    n_max: int,
) -> np.ndarray:
    """Geometric area per planning unit, padded to n_max.

    Returns a float32 array of shape (n_max,) in CRS units squared.
    Padded slots are 0.0.
    """
    if problem.n_planning_units > n_max:
        raise ValueError(
            f"n_max={n_max} is smaller than n_planning_units="
            f"{problem.n_planning_units}"
        )
    out = np.zeros(n_max, dtype=np.float32)
    areas = problem.planning_units.geometry.area.astype(float).to_numpy()
    out[: problem.n_planning_units] = areas
    return out


def build_eligibility_mask(
    problem: VectorConnectivityProblem,
    *,
    n_max: int,
) -> np.ndarray:
    """Boolean mask: True where the planning unit is eligible.

    Returns a bool array of shape (n_max,). Padded slots are False.
    """
    if problem.n_planning_units > n_max:
        raise ValueError(
            f"n_max={n_max} is smaller than n_planning_units="
            f"{problem.n_planning_units}"
        )
    out = np.zeros(n_max, dtype=bool)
    elig = problem.planning_units[problem.eligibility_column].astype(bool).to_numpy()
    out[: problem.n_planning_units] = elig
    return out
