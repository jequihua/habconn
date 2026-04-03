import numpy as np

from habconn.problems.candidate_generation import (
    CandidateGenerator,
    CandidateRankingStrategy,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


def make_problem() -> VectorConnectivityProblem:
    # This test assumes your real example data exists.
    # Good enough for now, since we are still building on the example dataset.
    from pathlib import Path

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


def test_candidate_generation_fixed_k() -> None:
    problem = make_problem()
    state = LandscapeState.initialize(problem, budget=2)

    generator = CandidateGenerator(
        k=5,
        ranking_strategy=CandidateRankingStrategy.BY_PU_ID,
        pad_value=-1,
    )

    candidate_set = generator.generate(problem, state)

    assert len(candidate_set.candidate_pu_ids) == 5
    assert candidate_set.valid_mask.shape == (5,)
    assert candidate_set.scores.shape == (5,)
    assert candidate_set.n_valid == 5


def test_candidate_generation_respects_budget_and_selection() -> None:
    problem = make_problem()
    state0 = LandscapeState.initialize(problem, budget=2)
    first_pu = problem.planning_unit_ids[0]
    state1 = state0.apply_action(problem, first_pu)

    generator = CandidateGenerator(
        k=5,
        ranking_strategy=CandidateRankingStrategy.BY_PU_ID,
        pad_value=-1,
    )

    candidate_set = generator.generate(problem, state1)

    assert first_pu not in candidate_set.valid_pu_ids()
