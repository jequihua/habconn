import numpy as np

from habconn.problems.candidate_generation import CandidateSet
from habconn.state.action_mapping import CandidateActionMapper


def test_action_mapping_valid_and_invalid() -> None:
    candidate_set = CandidateSet(
        candidate_pu_ids=[5, 7, -1, -1],
        valid_mask=np.array([True, True, False, False], dtype=bool),
        scores=np.array([1.0, 0.9, np.nan, np.nan], dtype=float),
        pad_value=-1,
        strategy="by_pu_id",
    )

    mapper = CandidateActionMapper(pad_value=-1)

    valid = mapper.map_action(1, candidate_set)
    invalid = mapper.map_action(2, candidate_set)

    assert valid.is_valid is True
    assert valid.pu_id == 7

    assert invalid.is_valid is False
    assert invalid.pu_id == -1