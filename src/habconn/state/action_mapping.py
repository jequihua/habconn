# Map policy action slots to real planning units.

from __future__ import annotations

from dataclasses import dataclass

from habconn.problems.candidate_generation import CandidateSet


@dataclass(slots=True)
class ActionMappingResult:
    action_index: int
    pu_id: int
    is_valid: bool

    def to_dict(self) -> dict:
        return {
            "action_index": self.action_index,
            "pu_id": self.pu_id,
            "is_valid": self.is_valid,
        }


class CandidateActionMapper:
    """
    Maps policy action slots in [0, K-1] to actual planning-unit IDs.

    This is the core bridge for Option B:
    - policy chooses an index in a fixed Discrete(K)
    - mapper resolves that index to a real pu_id or marks it invalid

    Future encoders / policies can change how actions are scored,
    but as long as they emit a slot index, this mapper can stay stable.
    """

    def __init__(self, *, pad_value: int = -1) -> None:
        self.pad_value = int(pad_value)

    def map_action(
        self,
        action_index: int,
        candidate_set: CandidateSet,
    ) -> ActionMappingResult:
        if action_index < 0 or action_index >= candidate_set.k:
            raise IndexError(
                f"Action index {action_index} is out of bounds for candidate set of size {candidate_set.k}."
            )

        pu_id = int(candidate_set.candidate_pu_ids[action_index])
        is_valid = bool(candidate_set.valid_mask[action_index]) and pu_id != self.pad_value

        return ActionMappingResult(
            action_index=int(action_index),
            pu_id=pu_id,
            is_valid=is_valid,
        )

    def require_valid_pu_id(
        self,
        action_index: int,
        candidate_set: CandidateSet,
    ) -> int:
        result = self.map_action(action_index, candidate_set)
        if not result.is_valid:
            raise ValueError(
                f"Action index {action_index} maps to an invalid/padded slot. "
                f"Candidate IDs: {candidate_set.candidate_pu_ids}, "
                f"valid_mask: {candidate_set.valid_mask.tolist()}"
            )
        return int(result.pu_id)
