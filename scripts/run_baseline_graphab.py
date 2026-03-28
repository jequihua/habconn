# scripts/run_baseline_graphab.py

from __future__ import annotations

from pathlib import Path

from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.evaluators.graphab_runner import (
    GraphabProjectConfig,
    GraphabRunner,
    GraphabRuntimeConfig,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState

JAVAPATH = r"C:\Program Files\Java\jdk-17\bin\java.exe"

def main() -> None:
    # ------------------------------------------------------------------
    # EDIT THESE PATHS FOR YOUR MACHINE / TEST DATA
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    vector_path = project_root / "data" / "examples" / "small_vector_001" / "candidates.shp"
    habitat_raster_path = project_root / "data" / "examples" / "small_vector_001" / "habitat.tif"
    resistance_raster_path = project_root / "data" / "examples" / "small_vector_001" / "resistance.tif"

    graphab_jar_path = project_root / "tools" / "graphab.jar"
    graphab_work_root = project_root / "tmp" / "graphab_runs"

    # ------------------------------------------------------------------
    # PROBLEM
    # ------------------------------------------------------------------
    problem = VectorConnectivityProblem.from_files(
        name="small_vector_001",
        vector_path=vector_path,
        habitat_raster_path=habitat_raster_path,
        resistance_raster_path=resistance_raster_path,
        id_column="lyr.1",
        area_column="area",
        use_area_as_cost=False,
        uniform_cost=1.0,
        restored_resistance_value=None,  # None => use minimum resistance value
        habitat_value=1,
        all_touched=False,
    )

    print("Problem summary:")
    for key, value in problem.summary().items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # GRAPHAB
    # ------------------------------------------------------------------
    runtime_cfg = GraphabRuntimeConfig(
        graphab_jar_path=graphab_jar_path,
        work_root=graphab_work_root,
        java_executable=JAVAPATH,
        keep_workdirs=True,
        headless=True,
        jvm_memory="4G",
    )

    # Graphab version 2.8
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

# Graphab version 3.0
#    project_cfg = GraphabProjectConfig(
#        nodata_value=-32768,
#        habitat_name="habitat_main",
#        habitat_codes=(1,),
#        minarea=None,
#        maxsize=10.0,
#        con8=True,
#        novoronoi=False,
#        linkset_name="linkset_main",
#        distance_type="cost",
#        max_cost=300.0,
#        topo="planar",
#        save_paths=False,
#        graph_name="graph_main",
#        graph_threshold=130.0,
#        cost_converted_threshold=True,
#        metric_name="PC",
#        metric_d=130.0,
#        metric_p=0.01,
#    )

    runner = GraphabRunner(runtime_cfg, project_cfg)
    evaluator = GraphabEvaluator(runner)

    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------
    state0 = LandscapeState.initialize(problem, budget=2)
    result0 = evaluator.evaluate(problem, state0, run_label="baseline")
    state0 = state0.with_pc_value(result0.pc_value)

    print("\nBaseline evaluation:")
    print(f"  selected_pu_ids: {state0.selected_pu_ids}")
    print(f"  remaining_budget: {state0.remaining_budget}")
    print(f"  PC: {result0.pc_value}")

    # ------------------------------------------------------------------
    # FIRST MANUAL ACTION
    # ------------------------------------------------------------------
    pu_ids = problem.planning_unit_ids
    if len(pu_ids) < 2:
        raise ValueError("Need at least 2 polygons in the test vector file.")

    state1 = state0.apply_action(problem, pu_ids[0])
    result1 = evaluator.evaluate(problem, state1, run_label="after_action_1")
    state1 = state1.with_pc_value(result1.pc_value)

    print("\nAfter action 1:")
    print(f"  selected_pu_ids: {state1.selected_pu_ids}")
    print(f"  remaining_budget: {state1.remaining_budget}")
    print(f"  PC: {result1.pc_value}")
    print(f"  delta_PC: {result1.pc_value - result0.pc_value}")

    # ------------------------------------------------------------------
    # SECOND MANUAL ACTION
    # ------------------------------------------------------------------
    state2 = state1.apply_action(problem, pu_ids[1])
    result2 = evaluator.evaluate(problem, state2, run_label="after_action_2")
    state2 = state2.with_pc_value(result2.pc_value)

    print("\nAfter action 2:")
    print(f"  selected_pu_ids: {state2.selected_pu_ids}")
    print(f"  remaining_budget: {state2.remaining_budget}")
    print(f"  PC: {result2.pc_value}")
    print(f"  delta_PC: {result2.pc_value - result1.pc_value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
