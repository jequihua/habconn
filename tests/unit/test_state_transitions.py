from pathlib import Path

from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def make_problem() -> VectorConnectivityProblem:
    project_root = Path(__file__).resolve().parents[2]
    return VectorConnectivityProblem.from_files(
        name="small_vector_001",
        vector_path=project_root / "data" / "examples" / "small_vector_001" / "candidates.shp",
        habitat_raster_path=project_root / "data" / "examples" / "small_vector_001" / "habitat.tif",
        resistance_raster_path=project_root / "data" / "examples" / "small_vector_001" / "resistance.tif",
        id_column="lyr_1",
        area_column="area",
        use_area_as_cost=False,
        uniform_cost=1.0,
    )


def test_apply_action_reduces_budget() -> None:
    problem = make_problem()
    state0 = LandscapeState.initialize(problem, budget=2)
    pu_id = problem.planning_unit_ids[0]

    state1 = state0.apply_action(problem, pu_id)

    assert state1.remaining_budget == 1.0
    assert pu_id in state1.selected_pu_ids
