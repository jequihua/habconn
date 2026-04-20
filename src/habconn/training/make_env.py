"""Factory functions to create training and evaluation environments.

Encapsulates the problem-loading + backend-creation + env-creation
pipeline so training code doesn't assemble these ad hoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from habconn.envs.vector_env import VectorHabitatEnv
from habconn.evaluators.cli_exact_backend import CliExactBackend
from habconn.evaluators.graphab_evaluator import GraphabEvaluator
from habconn.evaluators.graphab_runner import (
    GraphabProjectConfig,
    GraphabRunner,
    GraphabRuntimeConfig,
)
from habconn.problems.candidate_generation import CandidateRankingStrategy
from habconn.problems.vector_problem import VectorConnectivityProblem


def make_env(
    *,
    data_dir: Path,
    graphab_jar: Path,
    work_root: Path,
    budget: int = 3,
    k: int = 10,
    java_executable: str = "java",
    jvm_memory: str = "4G",
    ranking_strategy: CandidateRankingStrategy = CandidateRankingStrategy.BY_PU_ID,
    random_seed: Optional[int] = None,
    enable_cache: bool = True,
) -> VectorHabitatEnv:
    """Create a VectorHabitatEnv from data paths.

    This is the canonical env factory for training and evaluation.
    All problem/backend/env assembly lives here, not in scripts.
    """
    problem = VectorConnectivityProblem.from_files(
        name=data_dir.name,
        vector_path=data_dir / "candidates.shp",
        habitat_raster_path=data_dir / "habitat.tif",
        resistance_raster_path=data_dir / "resistance.tif",
        id_column="lyr_1",
        area_column="area",
        uniform_cost=1.0,
    )

    runtime_cfg = GraphabRuntimeConfig(
        graphab_jar_path=graphab_jar,
        work_root=work_root,
        java_executable=java_executable,
        keep_workdirs=False,
        jvm_memory=jvm_memory,
    )
    project_cfg = GraphabProjectConfig()
    runner = GraphabRunner(runtime_cfg, project_cfg)
    evaluator = GraphabEvaluator(runner, enable_cache=enable_cache)
    backend = CliExactBackend(evaluator)

    return VectorHabitatEnv(
        problem=problem,
        backend=backend,
        k=k,
        budget=budget,
        ranking_strategy=ranking_strategy,
        random_seed=random_seed,
    )
