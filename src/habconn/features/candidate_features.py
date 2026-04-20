"""Per-candidate-slot feature builders.

Candidate-level features are aligned with the fixed-K candidate-set slot
index: for each of K slots, they report values for the candidate at that
slot (or 0/False for padded slots).
"""

from __future__ import annotations

import numpy as np

from habconn.problems.candidate_generation import CandidateSet
from habconn.problems.vector_problem import VectorConnectivityProblem


def build_candidate_costs(
    problem: VectorConnectivityProblem,
    candidate_set: CandidateSet,
) -> np.ndarray:
    """Restoration cost per candidate slot.

    Returns a float32 array of shape (K,). Padded slots are 0.0.
    """
    k = candidate_set.k
    out = np.zeros(k, dtype=np.float32)
    for i, pu_id in enumerate(candidate_set.candidate_pu_ids):
        if candidate_set.valid_mask[i]:
            out[i] = float(problem.get_cost(pu_id))
    return out


def build_candidate_areas(
    problem: VectorConnectivityProblem,
    candidate_set: CandidateSet,
) -> np.ndarray:
    """Geometric area per candidate slot.

    Returns a float32 array of shape (K,) in CRS units squared.
    Padded slots are 0.0.
    """
    k = candidate_set.k
    out = np.zeros(k, dtype=np.float32)
    for i, pu_id in enumerate(candidate_set.candidate_pu_ids):
        if candidate_set.valid_mask[i]:
            row = problem.get_planning_unit_row(pu_id)
            out[i] = float(row.geometry.area)
    return out


def build_candidate_scores(candidate_set: CandidateSet) -> np.ndarray:
    """Ranking scores from the candidate generator.

    Returns a float32 array of shape (K,). Padded slots are 0.0
    (scores are NaN in the candidate set; replaced with 0 here so
    the observation is finite and the extractor can consume it cleanly).
    """
    k = candidate_set.k
    out = np.zeros(k, dtype=np.float32)
    valid_scores = np.where(candidate_set.valid_mask, candidate_set.scores, 0.0)
    out[:] = valid_scores.astype(np.float32)
    # Replace any leftover NaN (e.g. from generators that use NaN for padding)
    out = np.nan_to_num(out, nan=0.0)
    return out
