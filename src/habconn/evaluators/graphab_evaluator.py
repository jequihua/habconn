# src/habconn/evaluators/graphab_evaluator.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from habconn.evaluators.graphab_runner import (
    GraphabRunResult,
    GraphabRunner,
)
from habconn.problems.vector_problem import VectorConnectivityProblem
from habconn.state.landscape_state import LandscapeState


@dataclass(slots=True)
class GraphabEvaluationResult:
    pc_value: float
    selected_pu_ids: list[int]
    habitat_raster_path: Path
    resistance_raster_path: Path
    graphab_project_dir: Path
    metric_file_path: Path
    raw_metric_table: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class GraphabEvaluator:
    def __init__(self, runner: GraphabRunner) -> None:
        self.runner = runner

    def evaluate(
        self,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        *,
        run_label: Optional[str] = None,
    ) -> GraphabEvaluationResult:
        run_dir = self.runner.make_run_directory(prefix=run_label or "eval")

        try:
            habitat_out = run_dir / "habitat_modified.tif"
            resistance_out = run_dir / "resistance_modified.tif"

            self._materialize_modified_rasters(
                problem=problem,
                state=state,
                habitat_out=habitat_out,
                resistance_out=resistance_out,
            )

            run_result = self.runner.run_full_pipeline(
                habitat_raster_path=habitat_out,
                resistance_raster_path=resistance_out,
                run_label=run_label or "graphab_eval",
            )

            metric_df = self._read_metric_table(run_result)
            pc_value = self._extract_pc_value(metric_df)

            return GraphabEvaluationResult(
                pc_value=pc_value,
                selected_pu_ids=list(state.selected_pu_ids),
                habitat_raster_path=habitat_out,
                resistance_raster_path=resistance_out,
                graphab_project_dir=run_result.project_dir,
                metric_file_path=run_result.metric_file_path,
                raw_metric_table=metric_df,
                metadata={
                    "run_dir": str(run_dir),
                    "project_name": run_result.project_name,
                    "step_count": state.step_count,
                },
            )
        except Exception:
            if not self.runner.runtime_config.keep_workdirs:
                self.runner.cleanup_run_directory(run_dir)
            raise

    def _materialize_modified_rasters(
        self,
        *,
        problem: VectorConnectivityProblem,
        state: LandscapeState,
        habitat_out: Path,
        resistance_out: Path,
    ) -> None:
        with rasterio.open(problem.habitat_raster_path) as hab_src, rasterio.open(
            problem.resistance_raster_path
        ) as res_src:
            habitat_arr = hab_src.read(problem.habitat_band)
            resistance_arr = res_src.read(problem.resistance_band)

            selected_gdf = problem.selected_geodataframe(state.selected_pu_ids)

            if not selected_gdf.empty:
                burn_shapes = [(geom, 1) for geom in selected_gdf.geometry]

                burned_mask = rasterize(
                    burn_shapes,
                    out_shape=habitat_arr.shape,
                    transform=hab_src.transform,
                    fill=0,
                    all_touched=problem.all_touched,
                    dtype="uint8",
                ).astype(bool)

                habitat_arr = habitat_arr.copy()
                resistance_arr = resistance_arr.copy()

                habitat_arr[burned_mask] = int(problem.habitat_value)
                resistance_arr[burned_mask] = problem.restored_resistance_value

            # Force Graphab landscape raster to integer type for --create.
            habitat_arr = np.rint(habitat_arr).astype(np.int16, copy=False)

            habitat_profile = hab_src.profile.copy()
            resistance_profile = res_src.profile.copy()

            habitat_profile.update(count=1, dtype="int16", nodata=problem.habitat_nodata)
            resistance_profile.update(count=1)

            with rasterio.open(habitat_out, "w", **habitat_profile) as dst:
                dst.write(habitat_arr, 1)

            with rasterio.open(resistance_out, "w", **resistance_profile) as dst:
                dst.write(resistance_arr, 1)

    def _read_metric_table(self, run_result: GraphabRunResult) -> pd.DataFrame:
        metric_file = run_result.metric_file_path
        if not metric_file.exists():
            raise FileNotFoundError(
                f"Graphab metric file not found: {metric_file}\n"
                f"Pipeline stdout:\n{run_result.pipeline_result.stdout}\n"
                f"Pipeline stderr:\n{run_result.pipeline_result.stderr}"
            )

        return pd.read_csv(metric_file, sep="\t", header=0)

    def _extract_pc_value(self, metric_df: pd.DataFrame) -> float:
        lowered = {str(col).strip().lower(): col for col in metric_df.columns}

        if "pc" in lowered:
            series = pd.to_numeric(metric_df[lowered["pc"]], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

        numeric_df = metric_df.apply(pd.to_numeric, errors="coerce")
        numeric_values = numeric_df.to_numpy().flatten()
        numeric_values = numeric_values[~np.isnan(numeric_values)]

        if numeric_values.size == 0:
            raise ValueError(
                f"Could not extract a numeric PC value from metric table columns: {list(metric_df.columns)}"
            )

        return float(numeric_values[-1])