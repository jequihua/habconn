"""Global landscape and episode feature builders.

Global features are scalars describing the overall episode state.
They are returned as 1-element float32 arrays for Gymnasium compatibility.

Includes both raw and normalized versions of budget and selection counts
so the extractor can choose which to consume.
"""

from __future__ import annotations

import numpy as np

from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def build_global_features(
    problem: VectorConnectivityProblem,
    state: LandscapeState,
    *,
    current_pc: float,
    initial_budget: float,
) -> dict[str, np.ndarray]:
    """Build the global-scalar feature dict.

    Parameters
    ----------
    problem : VectorConnectivityProblem
    state : LandscapeState
    current_pc : float
        Most recent PC evaluation value.
    initial_budget : float
        Budget at reset, used to compute the fraction remaining.

    Returns
    -------
    dict[str, np.ndarray]
        Each value is a shape-(1,) float32 array.
    """
    n_pu = max(1, problem.n_planning_units)
    n_selected = len(state.selected_pu_ids)

    budget_remaining = np.array([state.remaining_budget], dtype=np.float32)
    budget_fraction = np.array(
        [state.remaining_budget / initial_budget if initial_budget > 0 else 0.0],
        dtype=np.float32,
    )
    step_count = np.array([state.step_count], dtype=np.int32)
    selected_fraction = np.array([n_selected / n_pu], dtype=np.float32)
    pc_array = np.array([current_pc], dtype=np.float32)

    return {
        "remaining_budget": budget_remaining,
        "budget_fraction": budget_fraction,
        "step_count": step_count,
        "selected_fraction": selected_fraction,
        "current_pc": pc_array,
    }
