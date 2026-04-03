from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


class CandidateRankingStrategy(str, Enum):
    """
    Initial candidate ranking strategies.

    These are intentionally simple and deterministic for v1.
    Later we can add richer ecological heuristics or learned proposals.
    """

    BY_PU_ID = "by_pu_id"
    LOWEST_COST_FIRST = "lowest_cost_first"
    HIGHEST_AREA_FIRST = "highest_area_first"
    RANDOM = "random"


@dataclass(slots=True)
class CandidateSet:
    """
    Fixed-K candidate action set for Option B.

    candidate_pu_ids:
        Length exactly K.
        Real candidates are positive integers.
        Padding slots are filled with pad_value.

    valid_mask:
        Boolean mask of length K.
        True where candidate_pu_ids refers to a real selectable planning unit.
        False where the slot is padding / invalid.

    scores:
        Optional heuristic scores aligned with candidate slots.
        Padding slots use np.nan.

    Notes:
    - K is fixed for the policy head.
    - the mapping from slot -> pu_id changes by state.
    """

    candidate_pu_ids: list[int]
    valid_mask: np.ndarray
    scores: np.ndarray
    pad_value: int
    strategy: str

    @property
    def k(self) -> int:
        return len(self.candidate_pu_ids)

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_mask))

    def valid_pu_ids(self) -> list[int]:
        return [
            int(pu_id)
            for pu_id, is_valid in zip(self.candidate_pu_ids, self.valid_mask)
            if is_valid
        ]

    def to_dict(self) -> dict:
        return {
            "candidate_pu_ids": list(self.candidate_pu_ids),
            "valid_mask": self.valid_mask.astype(bool).copy(),
            "scores": self.scores.astype(float).copy(),
            "pad_value": int(self.pad_value),
            "strategy": self.strategy,
        }


class CandidateGenerator:
    """
    Generates a fixed-size candidate set for Option B.

    Current logic:
    1. find all feasible planning units
    2. rank them with a simple deterministic strategy
    3. keep top-K
    4. pad to length K

    This is intentionally simple for v1, but the interface is designed so
    future implementations can replace only the ranking/proposal logic.
    """

    def __init__(
        self,
        *,
        k: int,
        ranking_strategy: CandidateRankingStrategy = CandidateRankingStrategy.BY_PU_ID,
        pad_value: int = -1,
        random_seed: Optional[int] = None,
        area_column_fallback: str = "area",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        self.k = int(k)
        self.ranking_strategy = ranking_strategy
        self.pad_value = int(pad_value)
        self.random_seed = random_seed
        self.area_column_fallback = area_column_fallback
        self._rng = np.random.default_rng(random_seed)

    def generate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
    ) -> CandidateSet:
        feasible_df = self._feasible_dataframe(problem, state)
        ranked_df = self._rank_candidates(problem, feasible_df)

        chosen_df = ranked_df.head(self.k).copy()

        candidate_pu_ids = chosen_df[problem.internal_id_column].astype(int).tolist()
        scores = chosen_df["_candidate_score"].astype(float).to_numpy()

        n_valid = len(candidate_pu_ids)
        n_pad = self.k - n_valid

        if n_pad > 0:
            candidate_pu_ids.extend([self.pad_value] * n_pad)
            scores = np.concatenate([scores, np.full(n_pad, np.nan, dtype=float)])

        valid_mask = np.zeros(self.k, dtype=bool)
        valid_mask[:n_valid] = True

        return CandidateSet(
            candidate_pu_ids=candidate_pu_ids,
            valid_mask=valid_mask,
            scores=scores,
            pad_value=self.pad_value,
            strategy=self.ranking_strategy.value,
        )

    def _feasible_dataframe(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
    ) -> pd.DataFrame:
        gdf = problem.planning_units.copy()

        selected_set = set(state.selected_pu_ids)

        gdf["_is_selected"] = gdf[problem.internal_id_column].isin(selected_set)
        gdf["_is_eligible"] = gdf[problem.eligibility_column].astype(bool)
        gdf["_cost_float"] = pd.to_numeric(gdf[problem.cost_column], errors="raise").astype(float)
        gdf["_within_budget"] = gdf["_cost_float"] <= float(state.remaining_budget)

        feasible = gdf.loc[
            (~gdf["_is_selected"]) & gdf["_is_eligible"] & gdf["_within_budget"]
        ].copy()

        return feasible

    def _rank_candidates(
        self,
        problem: VectorConnectivityProblem,
        feasible_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if feasible_df.empty:
            feasible_df["_candidate_score"] = pd.Series(dtype=float)
            return feasible_df

        strategy = self.ranking_strategy

        if strategy == CandidateRankingStrategy.BY_PU_ID:
            feasible_df["_candidate_score"] = (
                -feasible_df[problem.internal_id_column].astype(float)
            )
            feasible_df = feasible_df.sort_values(
                by=[problem.internal_id_column],
                ascending=True,
                kind="stable",
            )

        elif strategy == CandidateRankingStrategy.LOWEST_COST_FIRST:
            feasible_df["_candidate_score"] = -feasible_df["_cost_float"]
            feasible_df = feasible_df.sort_values(
                by=["_cost_float", problem.internal_id_column],
                ascending=[True, True],
                kind="stable",
            )

        elif strategy == CandidateRankingStrategy.HIGHEST_AREA_FIRST:
            area_col = self.area_column_fallback
            if area_col not in feasible_df.columns:
                raise ValueError(
                    f"Ranking strategy HIGHEST_AREA_FIRST requires column '{area_col}'."
                )
            feasible_df["_area_float"] = pd.to_numeric(
                feasible_df[area_col], errors="raise"
            ).astype(float)
            feasible_df["_candidate_score"] = feasible_df["_area_float"]
            feasible_df = feasible_df.sort_values(
                by=["_area_float", problem.internal_id_column],
                ascending=[False, True],
                kind="stable",
            )

        elif strategy == CandidateRankingStrategy.RANDOM:
            feasible_df["_candidate_score"] = self._rng.random(len(feasible_df))
            feasible_df = feasible_df.sort_values(
                by=["_candidate_score", problem.internal_id_column],
                ascending=[False, True],
                kind="stable",
            )

        else:
            raise ValueError(f"Unsupported ranking strategy: {strategy}")

        return feasible_df.reset_index(drop=True)
