"""Observation packing for the vector restoration environment.

Assembles the observation dictionary from problem, state, candidate set,
and evaluation results. Uses the dedicated feature builders in
`features/node_features.py`, `features/candidate_features.py`, and
`features/global_features.py` rather than computing features inline.

V2 observation keys (superset of v1):

    Action-level (K,):
        action_mask     : bool    — which candidate slots are valid
        candidate_ids   : int32   — planning-unit ID per slot (-1 = pad)
        candidate_costs : float32 — restoration cost per slot (0 = pad)
        candidate_areas : float32 — geometric area per slot (0 = pad)

    Node-level (N_max,):
        selected_mask      : bool    — already-selected planning units
        node_mask          : bool    — real vs padded planning-unit slots
        node_costs         : float32 — cost per planning unit (0 = pad)
        node_areas         : float32 — area per planning unit (0 = pad)
        eligibility_mask   : bool    — eligible planning units

    Global (1,):
        remaining_budget   : float32
        budget_fraction    : float32 — remaining / initial
        step_count         : int32
        selected_fraction  : float32 — n_selected / n_planning_units
        current_pc         : float32

All arrays are deterministic and stable across reset/step for a given
(problem, state, candidate_set).
"""

from __future__ import annotations

import numpy as np

from habconn.features.candidate_features import (
    build_candidate_areas,
    build_candidate_costs,
)
from habconn.features.global_features import build_global_features
from habconn.features.node_features import (
    build_eligibility_mask,
    build_node_areas,
    build_node_costs,
)
from habconn.problems.candidate_generation import CandidateSet
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState
from habconn.state.masks import build_action_mask, build_node_mask, build_selected_mask


def pack_observation(
    problem: VectorConnectivityProblem,
    state: LandscapeState,
    candidate_set: CandidateSet,
    *,
    n_max: int,
    current_pc: float = 0.0,
    initial_budget: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build the v2 observation dictionary.

    Parameters
    ----------
    problem : VectorConnectivityProblem
        Static problem definition.
    state : LandscapeState
        Current episode state.
    candidate_set : CandidateSet
        Current fixed-K candidate action set.
    n_max : int
        Maximum number of planning units (for padding).
    current_pc : float
        Most recent PC evaluation value.
    initial_budget : float
        Budget at reset, used to compute budget_fraction.

    Returns
    -------
    dict[str, np.ndarray]
        Observation dictionary with stable keys and shapes.
    """
    # Action-level features
    action_mask = build_action_mask(candidate_set)
    candidate_ids = np.array(candidate_set.candidate_pu_ids, dtype=np.int32)
    candidate_costs = build_candidate_costs(problem, candidate_set)
    candidate_areas = build_candidate_areas(problem, candidate_set)

    # Node-level features
    selected_mask = build_selected_mask(problem, state, n_max=n_max)
    node_mask = build_node_mask(problem, n_max=n_max)
    node_costs = build_node_costs(problem, n_max=n_max)
    node_areas = build_node_areas(problem, n_max=n_max)
    eligibility_mask = build_eligibility_mask(problem, n_max=n_max)

    # Global features
    globals_ = build_global_features(
        problem, state,
        current_pc=current_pc,
        initial_budget=initial_budget,
    )

    return {
        # Action-level
        "action_mask": action_mask,
        "candidate_ids": candidate_ids,
        "candidate_costs": candidate_costs,
        "candidate_areas": candidate_areas,
        # Node-level
        "selected_mask": selected_mask,
        "node_mask": node_mask,
        "node_costs": node_costs,
        "node_areas": node_areas,
        "eligibility_mask": eligibility_mask,
        # Global
        "remaining_budget": globals_["remaining_budget"],
        "budget_fraction": globals_["budget_fraction"],
        "step_count": globals_["step_count"],
        "selected_fraction": globals_["selected_fraction"],
        "current_pc": globals_["current_pc"],
    }
