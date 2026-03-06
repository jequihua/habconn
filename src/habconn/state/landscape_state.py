# src/habconn/state/landscape_state.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from habconn.problems.vector_problem import VectorConnectivityProblem


@dataclass(slots=True)
class LandscapeState:
    """
    Dynamic state for one optimization episode.

    Version 1:
    - selected planning units are tracked by pu_id
    - budget can be integer-like now, but is stored as float for future flexibility
    - cached_pc_value is the exact PC returned by Graphab for the current state
    """

    problem_name: str
    remaining_budget: float
    selected_pu_ids: list[int] = field(default_factory=list)
    step_count: int = 0
    cached_pc_value: Optional[float] = None
    done: bool = False
    info: dict = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        problem: VectorConnectivityProblem,
        *,
        budget: Optional[float] = None,
    ) -> "LandscapeState":
        if budget is None:
            budget = problem.default_budget
        return cls(
            problem_name=problem.name,
            remaining_budget=float(budget),
            selected_pu_ids=[],
            step_count=0,
            cached_pc_value=None,
            done=False,
            info={},
        )

    def is_selected(self, pu_id: int) -> bool:
        return pu_id in self.selected_pu_ids

    def can_select(self, problem: VectorConnectivityProblem, pu_id: int) -> bool:
        if self.done:
            return False

        if self.is_selected(pu_id):
            return False

        row = problem.get_planning_unit_row(pu_id)
        eligible = bool(row[problem.eligibility_column])
        if not eligible:
            return False

        cost = float(row[problem.cost_column])
        return self.remaining_budget >= cost

    def feasible_actions(self, problem: VectorConnectivityProblem) -> list[int]:
        feasible: list[int] = []
        for pu_id in problem.planning_unit_ids:
            if self.can_select(problem, pu_id):
                feasible.append(pu_id)
        return feasible

    def apply_action(self, problem: VectorConnectivityProblem, pu_id: int) -> "LandscapeState":
        if not self.can_select(problem, pu_id):
            raise ValueError(
                f"Invalid action: cannot select pu_id={pu_id}. "
                f"Remaining budget={self.remaining_budget}, selected={self.selected_pu_ids}"
            )

        cost = problem.get_cost(pu_id)

        new_selected = list(self.selected_pu_ids)
        new_selected.append(int(pu_id))

        new_state = LandscapeState(
            problem_name=self.problem_name,
            remaining_budget=float(self.remaining_budget - cost),
            selected_pu_ids=new_selected,
            step_count=self.step_count + 1,
            cached_pc_value=None,
            done=False,
            info=dict(self.info),
        )

        if len(new_state.feasible_actions(problem)) == 0:
            new_state.done = True

        return new_state

    def with_pc_value(self, pc_value: float) -> "LandscapeState":
        return LandscapeState(
            problem_name=self.problem_name,
            remaining_budget=self.remaining_budget,
            selected_pu_ids=list(self.selected_pu_ids),
            step_count=self.step_count,
            cached_pc_value=float(pc_value),
            done=self.done,
            info=dict(self.info),
        )

    def to_dict(self) -> dict:
        return {
            "problem_name": self.problem_name,
            "remaining_budget": self.remaining_budget,
            "selected_pu_ids": list(self.selected_pu_ids),
            "step_count": self.step_count,
            "cached_pc_value": self.cached_pc_value,
            "done": self.done,
            "info": dict(self.info),
        }
