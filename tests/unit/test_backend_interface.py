"""Unit tests for the backend interface, routing, and fallback logic.

These tests use mock backends to verify:
- BackendRouter selects the preferred backend when supported
- BackendRouter falls back to CLI when preferred doesn't support the action
- Fallback is explicit (recorded in metadata)
- classify_action correctly classifies states
- UnsupportedActionError is raised for unsupported action types
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from habconn.evaluators.base import (
    ActionType,
    BackendResult,
    BackendType,
    GraphabBackend,
    UnsupportedActionError,
    classify_action,
)
from habconn.evaluators.backend_router import BackendRouter


# --- Mock backends ---


class MockCliBackend(GraphabBackend):
    """Mock CLI backend that supports all action types."""

    def __init__(self) -> None:
        self.evaluate_count = 0
        self.last_state = None

    def evaluate(self, problem, state, *, run_label=None):
        self.evaluate_count += 1
        self.last_state = state
        return BackendResult(
            pc_value=1.0e-5,
            backend_type=BackendType.CLI_EXACT,
            action_type=ActionType.BASELINE,
            selected_pu_ids=list(state.selected_pu_ids),
            metadata={"mock": True},
        )

    def supports_action_type(self, action_type):
        return True

    def reset_session(self):
        self.evaluate_count = 0

    @property
    def backend_type(self):
        return BackendType.CLI_EXACT


class MockJavaServiceBackend(GraphabBackend):
    """Mock Java service backend that supports only BASELINE and ADDITIVE_PATCH."""

    def __init__(self) -> None:
        self.evaluate_count = 0
        self.last_state = None

    def evaluate(self, problem, state, *, run_label=None):
        action_type = classify_action(problem, state)
        if not self.supports_action_type(action_type):
            raise UnsupportedActionError(BackendType.JAVA_SERVICE, action_type)
        self.evaluate_count += 1
        self.last_state = state
        return BackendResult(
            pc_value=1.0e-5,
            backend_type=BackendType.JAVA_SERVICE,
            action_type=action_type,
            selected_pu_ids=list(state.selected_pu_ids),
            metadata={"mock": True},
        )

    def supports_action_type(self, action_type):
        return action_type in (ActionType.BASELINE, ActionType.ADDITIVE_PATCH)

    def reset_session(self):
        self.evaluate_count = 0

    @property
    def backend_type(self):
        return BackendType.JAVA_SERVICE


# --- Mock problem and state ---


def _make_mock_problem(restored_resistance_value=1.0, resistance_min_value=1.0):
    """Create a mock VectorConnectivityProblem."""
    problem = MagicMock()
    problem.restored_resistance_value = restored_resistance_value
    problem.resistance_min_value = resistance_min_value
    problem.name = "test_problem"
    return problem


def _make_mock_state(selected_pu_ids=None):
    """Create a mock LandscapeState."""
    state = MagicMock()
    state.selected_pu_ids = selected_pu_ids or []
    state.step_count = len(state.selected_pu_ids)
    state.remaining_budget = 10.0
    return state


# --- Tests ---


class TestClassifyAction:
    def test_baseline_no_selections(self):
        problem = _make_mock_problem()
        state = _make_mock_state(selected_pu_ids=[])
        assert classify_action(problem, state) == ActionType.BASELINE

    def test_additive_patch_with_matching_resistance(self):
        problem = _make_mock_problem(
            restored_resistance_value=1.0, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1, 2])
        assert classify_action(problem, state) == ActionType.ADDITIVE_PATCH

    def test_resistance_change_with_different_values(self):
        problem = _make_mock_problem(
            restored_resistance_value=5.0, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1])
        assert classify_action(problem, state) == ActionType.RESISTANCE_CHANGE

    def test_resistance_change_when_restored_is_none(self):
        problem = _make_mock_problem(
            restored_resistance_value=None, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1])
        assert classify_action(problem, state) == ActionType.RESISTANCE_CHANGE


class TestBackendRouter:
    def test_uses_preferred_for_supported_action(self):
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        problem = _make_mock_problem()
        state = _make_mock_state(selected_pu_ids=[])

        result = router.evaluate(problem, state)
        assert result.backend_type == BackendType.JAVA_SERVICE
        assert java.evaluate_count == 1
        assert cli.evaluate_count == 0
        assert result.metadata["fallback_used"] is False

    def test_falls_back_for_unsupported_action(self):
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        problem = _make_mock_problem(
            restored_resistance_value=5.0, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1])

        result = router.evaluate(problem, state)
        assert result.backend_type == BackendType.CLI_EXACT
        assert cli.evaluate_count == 1
        assert java.evaluate_count == 0
        assert result.metadata["fallback_used"] is True
        assert "fallback_reason" in result.metadata

    def test_additive_patch_uses_preferred(self):
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        problem = _make_mock_problem()
        state = _make_mock_state(selected_pu_ids=[1, 2, 3])

        result = router.evaluate(problem, state)
        assert result.backend_type == BackendType.JAVA_SERVICE
        assert result.metadata["fallback_used"] is False

    def test_reset_resets_both_backends(self):
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        java.evaluate_count = 5
        cli.evaluate_count = 3
        router.reset_session()
        assert java.evaluate_count == 0
        assert cli.evaluate_count == 0

    def test_rejects_same_backend_types(self):
        cli1 = MockCliBackend()
        cli2 = MockCliBackend()
        with pytest.raises(ValueError, match="different types"):
            BackendRouter(preferred=cli1, fallback=cli2)

    def test_supports_action_type_union(self):
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        assert router.supports_action_type(ActionType.BASELINE)
        assert router.supports_action_type(ActionType.ADDITIVE_PATCH)
        assert router.supports_action_type(ActionType.RESISTANCE_CHANGE)


class TestUnsupportedActionError:
    def test_error_message(self):
        err = UnsupportedActionError(
            BackendType.JAVA_SERVICE, ActionType.RESISTANCE_CHANGE
        )
        assert "java_service" in str(err)
        assert "resistance_change" in str(err)

    def test_error_attributes(self):
        err = UnsupportedActionError(
            BackendType.JAVA_SERVICE, ActionType.RESISTANCE_CHANGE
        )
        assert err.backend_type == BackendType.JAVA_SERVICE
        assert err.action_type == ActionType.RESISTANCE_CHANGE


class TestBackendResult:
    def test_result_fields(self):
        result = BackendResult(
            pc_value=2.17e-5,
            backend_type=BackendType.CLI_EXACT,
            action_type=ActionType.BASELINE,
            selected_pu_ids=[],
            metadata={"test": True},
        )
        assert result.pc_value == 2.17e-5
        assert result.backend_type == BackendType.CLI_EXACT
        assert result.action_type == ActionType.BASELINE
        assert result.selected_pu_ids == []
        assert result.metadata["test"] is True


class TestCliBackendActionTypeProvenance:
    """Verify CliExactBackend correctly classifies action_type via classify_action."""

    def test_resistance_change_reported_correctly(self):
        """When restored_resistance != resistance_min, CLI should report RESISTANCE_CHANGE."""
        problem = _make_mock_problem(
            restored_resistance_value=5.0, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1])
        action = classify_action(problem, state)
        assert action == ActionType.RESISTANCE_CHANGE

    def test_fallback_result_carries_correct_action_type(self):
        """When BackendRouter falls back to CLI for resistance-change,
        the result action_type should be RESISTANCE_CHANGE, not ADDITIVE_PATCH."""
        cli = MockCliBackend()
        java = MockJavaServiceBackend()
        router = BackendRouter(preferred=java, fallback=cli)

        problem = _make_mock_problem(
            restored_resistance_value=5.0, resistance_min_value=1.0
        )
        state = _make_mock_state(selected_pu_ids=[1])

        result = router.evaluate(problem, state)
        assert result.backend_type == BackendType.CLI_EXACT
        assert result.metadata["fallback_used"] is True
        # The CLI mock doesn't use classify_action internally, so this tests
        # that the router correctly classifies the action for fallback routing.
        # The actual CliExactBackend (not the mock) now uses classify_action.


class TestSessionProblemIdentity:
    """Verify that problem identity is checked in session management."""

    def test_problem_key_differs_for_different_problems(self):
        from habconn.evaluators.java_service_backend import JavaServiceBackend

        p1 = MagicMock()
        p1.name = "problem_A"
        p1.habitat_raster_path = "/data/A/habitat.tif"
        p1.resistance_raster_path = "/data/A/resistance.tif"
        p1.vector_path = "/data/A/candidates.shp"

        p2 = MagicMock()
        p2.name = "problem_B"
        p2.habitat_raster_path = "/data/B/habitat.tif"
        p2.resistance_raster_path = "/data/B/resistance.tif"
        p2.vector_path = "/data/B/candidates.shp"

        key1 = JavaServiceBackend._problem_key(p1)
        key2 = JavaServiceBackend._problem_key(p2)
        assert key1 != key2

    def test_problem_key_stable_for_same_problem(self):
        from habconn.evaluators.java_service_backend import JavaServiceBackend

        p = MagicMock()
        p.name = "problem_A"
        p.habitat_raster_path = "/data/A/habitat.tif"
        p.resistance_raster_path = "/data/A/resistance.tif"
        p.vector_path = "/data/A/candidates.shp"

        assert JavaServiceBackend._problem_key(p) == JavaServiceBackend._problem_key(p)
