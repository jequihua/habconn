"""Backend routing: selects between backends and handles fallback.

The router uses the preferred backend when it supports the action type,
and falls back to the CLI exact backend otherwise. Fallback is always
explicit and logged.
"""

from __future__ import annotations

import logging
from typing import Optional

from habconn.evaluators.base import (
    BackendResult,
    BackendType,
    GraphabBackend,
    UnsupportedActionError,
    classify_action,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState

logger = logging.getLogger(__name__)


class BackendRouter(GraphabBackend):
    """Routes evaluations to the appropriate backend with explicit fallback.

    If the preferred backend does not support the action type implied by
    the current state, the router falls back to the CLI exact backend.
    Fallback is never silent: it is logged and recorded in metadata.
    """

    def __init__(
        self,
        preferred: GraphabBackend,
        fallback: GraphabBackend,
    ) -> None:
        if preferred.backend_type == fallback.backend_type:
            raise ValueError(
                "Preferred and fallback backends must be different types"
            )
        self._preferred = preferred
        self._fallback = fallback

    def evaluate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        *,
        run_label: Optional[str] = None,
    ) -> BackendResult:
        action_type = classify_action(problem, state)

        if self._preferred.supports_action_type(action_type):
            try:
                result = self._preferred.evaluate(
                    problem, state, run_label=run_label,
                )
                result.metadata["fallback_used"] = False
                return result
            except UnsupportedActionError:
                pass  # Fall through to fallback

        # Explicit fallback
        logger.info(
            "Falling back to %s for action_type=%s (preferred=%s does not support it)",
            self._fallback.backend_type.value,
            action_type.value,
            self._preferred.backend_type.value,
        )

        result = self._fallback.evaluate(problem, state, run_label=run_label)
        result.metadata["fallback_used"] = True
        result.metadata["fallback_reason"] = (
            f"{self._preferred.backend_type.value} does not support {action_type.value}"
        )
        return result

    def supports_action_type(self, action_type: "ActionType") -> bool:
        return (
            self._preferred.supports_action_type(action_type)
            or self._fallback.supports_action_type(action_type)
        )

    def reset_session(self) -> None:
        self._preferred.reset_session()
        self._fallback.reset_session()

    @property
    def backend_type(self) -> BackendType:
        return self._preferred.backend_type

    @property
    def preferred(self) -> GraphabBackend:
        return self._preferred

    @property
    def fallback(self) -> GraphabBackend:
        return self._fallback
