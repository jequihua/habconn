from __future__ import annotations

from pathlib import Path
import math

import pytest

from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.evaluators.graphab_runner import (
    GraphabProjectConfig,
    GraphabRunner,
    GraphabRuntimeConfig,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


@pytest.mark.integration
def test_graphab_evaluator_smoke() -> None:
    project_root = Path(__file__).resolve().parents[2]

    vector_path = project_root / "data" / "examples" / "small_vector_001" / "candidates.shp"
    habitat_raster_path = project_root / "data" / "examples" / "small_vector_001" / "habitat.tif"
    resistance_raster_path = project_root / "data" / "examples" / "small_vector_001" / "resistance.tif"
    graphab_jar_path = project_root / "tools" / "graphab.jar"
    graphab_work_root = project_root / "tmp" / "graphab_test_runs"

    if not vector_path.exists():
        pytest.skip("Example vector data not found.")
    if not habitat_raster_path.exists():
        pytest.skip("Example habitat raster not found.")
    if not resistance_raster_path.exists():
        pytest.skip("Example resistance raster not found.")
    if not graphab_jar_path.exists():
        pytest.skip("Graphab jar not found.")

    problem = VectorConnectivityProblem.from_files(
        name="small_vector_001",
        vector_path=vector_path,
        habitat_raster_path=habitat_raster_path,
        resistance_raster_path=resistance_raster_path,
        id_column="lyr_1",
        area_column="area",
        use_area_as_cost=False,
        uniform_cost=1.0,
        restored_resistance_value=None,
        habitat_value=1,
        all_touched=False,
    )

    runtime_cfg = GraphabRuntimeConfig(
        graphab_jar_path=graphab_jar_path,
        work_root=graphab_work_root,
        java_executable="java",
        keep_workdirs=False,
        headless=True,
        jvm_memory="4G",
        capture_output=True,
        max_log_lines_on_error=40,
    )

    project_cfg = GraphabProjectConfig(
        habitat_codes=(1,),
        nodata_value=-32768,
        minarea=None,
        maxsize=10.0,
        con8=True,
        linkset_name="linkset_main",
        distance_type="cost",
        complete=False,
        max_cost=300.0,
        save_paths=False,
        graph_name="graph_main",
        graph_threshold=130.0,
        cost_converted_threshold=True,
        nointra=False,
        metric_name="PC",
        metric_d=130.0,
        metric_p=0.01,
        metric_beta=None,
    )

    runner = GraphabRunner(runtime_cfg, project_cfg)
    evaluator = GraphabEvaluator(runner, enable_cache=True)

    state0 = LandscapeState.initialize(problem, budget=2)
    result0 = evaluator.evaluate(problem, state0, run_label="pytest_baseline")

    assert math.isfinite(result0.pc_value)
    assert result0.pc_value >= 0.0

    pu_ids = problem.planning_unit_ids
    assert len(pu_ids) >= 1

    state1 = state0.apply_action(problem, pu_ids[0])
    result1 = evaluator.evaluate(problem, state1, run_label="pytest_after_action_1")

    assert math.isfinite(result1.pc_value)
    assert result1.pc_value >= 0.0