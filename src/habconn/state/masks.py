# Action and node masking logic.

from __future__ import annotations

import numpy as np

from habconn.problems.candidate_generation import CandidateSet
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def build_action_mask(candidate_set: CandidateSet) -> np.ndarray:
    """
    Build the fixed-K action mask for the policy.

    True  = valid action slot
    False = padded / invalid slot
    """
    return candidate_set.valid_mask.astype(bool).copy()


def build_node_mask(
    problem: VectorConnectivityProblem,
    *,
    n_max: int,
) -> np.ndarray:
    """
    Build a padded node mask for future observation packing.

    This is not yet tied to a full observation builder, but we add it now
    because it is part of the stable contract we want for future set/GNN models.

    True  = real planning unit row
    False = padded row
    """
    if n_max <= 0:
        raise ValueError(f"n_max must be > 0, got {n_max}")

    n_nodes = problem.n_planning_units
    if n_nodes > n_max:
        raise ValueError(
            f"Problem has {n_nodes} planning units, but n_max={n_max}. "
            "Increase n_max or introduce observation-side candidate truncation later."
        )

    mask = np.zeros(n_max, dtype=bool)
    mask[:n_nodes] = True
    return mask


def build_selected_mask(
    problem: VectorConnectivityProblem,
    state: LandscapeState,
    *,
    n_max: int,
) -> np.ndarray:
    """
    Build a padded selected/not-selected mask aligned with the planning-unit order.

    True  = planning unit is selected
    False = not selected or padded row
    """
    if n_max <= 0:
        raise ValueError(f"n_max must be > 0, got {n_max}")

    pu_ids = problem.planning_unit_ids
    if len(pu_ids) > n_max:
        raise ValueError(
            f"Problem has {len(pu_ids)} planning units, but n_max={n_max}."
        )

    selected = set(state.selected_pu_ids)
    mask = np.zeros(n_max, dtype=bool)

    for i, pu_id in enumerate(pu_ids):
        mask[i] = pu_id in selected

    return mask
