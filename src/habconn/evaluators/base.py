"""Abstract backend contract for Graphab connectivity evaluation.

This module defines the backend interface that the evaluator layer uses
to compute exact PC values. All Graphab-specific execution details
(CLI subprocess, Java service, etc.) are encapsulated behind this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


class BackendType(str, Enum):
    """Identifies which backend produced a result."""

    CLI_EXACT = "cli_exact"
    JAVA_SERVICE = "java_service"


class ActionType(str, Enum):
    """Classifies the type of landscape modification an evaluation requires."""

    BASELINE = "baseline"
    ADDITIVE_PATCH = "additive_patch"
    RESISTANCE_CHANGE = "resistance_change"


@dataclass(slots=True)
class BackendResult:
    """Result returned by any GraphabBackend implementation.

    Every backend must populate at least pc_value, backend_type,
    and action_type so that callers and tests can identify provenance.
    """

    pc_value: float
    backend_type: BackendType
    action_type: ActionType
    selected_pu_ids: list[int]
    metadata: dict = field(default_factory=dict)


class GraphabBackend(ABC):
    """Abstract contract for Graphab connectivity evaluation backends.

    Implementations must:
    - Accept a problem definition and landscape state
    - Return an exact PC value with provenance metadata
    - Declare which action types they support
    - Raise UnsupportedActionError for unsupported cases
    """

    @abstractmethod
    def evaluate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        *,
        run_label: Optional[str] = None,
    ) -> BackendResult:
        """Compute exact PC for the given problem and state."""

    @abstractmethod
    def supports_action_type(self, action_type: ActionType) -> bool:
        """Return True if this backend can handle the given action type."""

    @abstractmethod
    def reset_session(self) -> None:
        """Reset any persistent session state.

        For stateless backends (CLI), this is a no-op.
        For stateful backends (Java service), this clears the
        in-memory project and forces reload on next evaluate().
        """

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the type identifier for this backend."""


class UnsupportedActionError(Exception):
    """Raised when a backend cannot handle the requested action type."""

    def __init__(self, backend_type: BackendType, action_type: ActionType) -> None:
        self.backend_type = backend_type
        self.action_type = action_type
        super().__init__(
            f"Backend {backend_type.value} does not support "
            f"action type {action_type.value}"
        )


def classify_action(
    problem: VectorConnectivityProblem,
    state: LandscapeState,
) -> ActionType:
    """Classify the action type implied by the current state.

    - BASELINE: no planning units selected
    - ADDITIVE_PATCH: only additive restoration (habitat added, resistance
      set to restored_resistance_value which equals the raster minimum)
    - RESISTANCE_CHANGE: any modification that implies non-trivial
      resistance raster changes beyond simple patch addition

    For v1, we conservatively classify as ADDITIVE_PATCH when selected
    units exist and restored_resistance_value equals resistance_min_value.
    Otherwise we classify as RESISTANCE_CHANGE.
    """
    if not state.selected_pu_ids:
        return ActionType.BASELINE

    if (
        problem.restored_resistance_value is not None
        and problem.resistance_min_value is not None
        and problem.restored_resistance_value == problem.resistance_min_value
    ):
        return ActionType.ADDITIVE_PATCH

    return ActionType.RESISTANCE_CHANGE
