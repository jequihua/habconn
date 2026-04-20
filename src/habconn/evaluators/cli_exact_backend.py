"""CLI exact backend: wraps the current recreate-from-raster Graphab CLI pipeline.

This is the canonical exact reference backend. All other backends must
be validated against it before they are considered correct.
"""

from __future__ import annotations

from typing import Optional

from habconn.evaluators.base import (
    ActionType,
    BackendResult,
    BackendType,
    GraphabBackend,
    classify_action,
)
from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


class CliExactBackend(GraphabBackend):
    """Canonical exact reference backend using the Graphab CLI.

    Delegates to the existing GraphabEvaluator which materializes
    modified rasters and runs the full Graphab CLI pipeline per evaluation.

    This backend supports all action types (baseline, additive patch,
    resistance change) because it always recreates from rasters.
    """

    def __init__(self, evaluator: GraphabEvaluator) -> None:
        self._evaluator = evaluator

    def evaluate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        *,
        run_label: Optional[str] = None,
    ) -> BackendResult:
        label = run_label or "cli_exact"
        result = self._evaluator.evaluate(problem, state, run_label=label)

        action_type = classify_action(problem, state)

        return BackendResult(
            pc_value=result.pc_value,
            backend_type=BackendType.CLI_EXACT,
            action_type=action_type,
            selected_pu_ids=list(state.selected_pu_ids),
            metadata={
                "graphab_run_dir": str(result.graphab_run_dir),
                "graphab_project_dir": str(result.graphab_project_dir),
                "metric_file_path": str(result.metric_file_path),
                "step_count": state.step_count,
                "cached": result.metadata.get("cached", False),
            },
        )

    def supports_action_type(self, action_type: ActionType) -> bool:
        return True  # CLI recreates everything from rasters

    def reset_session(self) -> None:
        self._evaluator.clear_cache()

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CLI_EXACT
