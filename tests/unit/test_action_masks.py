from habconn.problems.candidate_generation import CandidateSet
from habconn.state.masks import build_action_mask
import numpy as np


def test_build_action_mask() -> None:
    candidate_set = CandidateSet(
        candidate_pu_ids=[10, 11, -1, -1],
        valid_mask=np.array([True, True, False, False], dtype=bool),
        scores=np.array([0.5, 0.4, np.nan, np.nan], dtype=float),
        pad_value=-1,
        strategy="by_pu_id",
    )

    mask = build_action_mask(candidate_set)

    assert mask.tolist() == [True, True, False, False]
