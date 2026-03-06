# src/habconn/problems/vector_problem.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


PathLike = Union[str, Path]


@dataclass(slots=True)
class VectorConnectivityProblem:
    """
    Static definition of one vector-action habitat connectivity problem.

    Version 1 assumptions:
    - actions select polygons from a vector layer
    - selected polygons become habitat
    - selected polygons also receive a restored resistance value
    - all polygons are initially eligible unless specified otherwise
    - costs may be uniform or read from a column
    """

    name: str
    vector_path: Path
    habitat_raster_path: Path
    resistance_raster_path: Path

    planning_units: gpd.GeoDataFrame

    id_column: str
    internal_id_column: str
    cost_column: str
    eligibility_column: str

    habitat_band: int = 1
    resistance_band: int = 1
    habitat_value: int = 1
    restored_resistance_value: Optional[float] = None
    all_touched: bool = False

    raster_crs: Optional[str] = None
    raster_width: int = 0
    raster_height: int = 0
    raster_transform: object = None
    habitat_nodata: Optional[float] = None
    resistance_nodata: Optional[float] = None
    habitat_dtype: str = ""
    resistance_dtype: str = ""
    resistance_min_value: Optional[float] = None
    resistance_max_value: Optional[float] = None

    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        *,
        name: str,
        vector_path: PathLike,
        habitat_raster_path: PathLike,
        resistance_raster_path: PathLike,
        id_column: str = "lyr.1",
        area_column: str = "area",
        internal_id_column: str = "pu_id",
        eligibility_column: str = "eligible",
        cost_column: str = "cost",
        use_area_as_cost: bool = False,
        uniform_cost: Optional[float] = 1.0,
        restored_resistance_value: Optional[float] = None,
        habitat_band: int = 1,
        resistance_band: int = 1,
        habitat_value: int = 1,
        all_touched: bool = False,
    ) -> "VectorConnectivityProblem":
        vector_path = Path(vector_path).expanduser().resolve()
        habitat_raster_path = Path(habitat_raster_path).expanduser().resolve()
        resistance_raster_path = Path(resistance_raster_path).expanduser().resolve()

        gdf = cls._read_vector(vector_path)

        if gdf.empty:
            raise ValueError(f"Vector file has no features: {vector_path}")

        if "geometry" not in gdf.columns:
            raise ValueError("Vector file does not contain a geometry column.")

        gdf = gdf[gdf.geometry.notna()].copy()
        if gdf.empty:
            raise ValueError("Vector file contains no valid geometries.")

        if id_column not in gdf.columns:
            raise ValueError(
                f"Required id column '{id_column}' not found in vector file columns: {list(gdf.columns)}"
            )

        if internal_id_column in gdf.columns:
            raise ValueError(
                f"Internal id column '{internal_id_column}' already exists in vector file. "
                "Choose a different internal_id_column name."
            )

        with rasterio.open(habitat_raster_path) as hab_src, rasterio.open(resistance_raster_path) as res_src:
            cls._validate_raster_alignment(hab_src, res_src)

            if gdf.crs is None:
                raise ValueError("Vector file has no CRS defined.")

            if gdf.crs != hab_src.crs:
                gdf = gdf.to_crs(hab_src.crs)

            gdf[internal_id_column] = np.arange(1, len(gdf) + 1, dtype=np.int64)

            if use_area_as_cost:
                if area_column not in gdf.columns:
                    raise ValueError(
                        f"use_area_as_cost=True but area column '{area_column}' was not found."
                    )
                gdf[cost_column] = pd.to_numeric(gdf[area_column], errors="raise").astype(float)
            else:
                if uniform_cost is None:
                    raise ValueError(
                        "uniform_cost must be provided when use_area_as_cost=False."
                    )
                gdf[cost_column] = float(uniform_cost)

            gdf[eligibility_column] = True

            resistance_arr = res_src.read(resistance_band)
            valid_res = resistance_arr
            if res_src.nodata is not None:
                valid_res = valid_res[valid_res != res_src.nodata]

            if valid_res.size == 0:
                resistance_min_value = None
                resistance_max_value = None
            else:
                resistance_min_value = float(np.min(valid_res))
                resistance_max_value = float(np.max(valid_res))

            final_restored_resistance = (
                resistance_min_value
                if restored_resistance_value is None
                else float(restored_resistance_value)
            )

            return cls(
                name=name,
                vector_path=vector_path,
                habitat_raster_path=habitat_raster_path,
                resistance_raster_path=resistance_raster_path,
                planning_units=gdf,
                id_column=id_column,
                internal_id_column=internal_id_column,
                cost_column=cost_column,
                eligibility_column=eligibility_column,
                habitat_band=habitat_band,
                resistance_band=resistance_band,
                habitat_value=habitat_value,
                restored_resistance_value=final_restored_resistance,
                all_touched=all_touched,
                raster_crs=str(hab_src.crs) if hab_src.crs is not None else None,
                raster_width=hab_src.width,
                raster_height=hab_src.height,
                raster_transform=hab_src.transform,
                habitat_nodata=hab_src.nodata,
                resistance_nodata=res_src.nodata,
                habitat_dtype=str(hab_src.dtypes[habitat_band - 1]),
                resistance_dtype=str(res_src.dtypes[resistance_band - 1]),
                resistance_min_value=resistance_min_value,
                resistance_max_value=resistance_max_value,
                metadata={
                    "area_column": area_column,
                    "source_feature_count": len(gdf),
                    "vector_driver": vector_path.suffix.lower(),
                },
            )

    @staticmethod
    def _read_vector(vector_path: Path) -> gpd.GeoDataFrame:
        suffix = vector_path.suffix.lower()
        if suffix not in {".shp", ".gpkg"}:
            raise ValueError(
                f"Unsupported vector format '{suffix}'. Supported now: .shp, .gpkg"
            )
        return gpd.read_file(vector_path)

    @staticmethod
    def _validate_raster_alignment(hab_src, res_src) -> None:
        if hab_src.crs != res_src.crs:
            raise ValueError(
                f"Habitat and resistance rasters have different CRS: "
                f"{hab_src.crs} vs {res_src.crs}"
            )

        if hab_src.transform != res_src.transform:
            raise ValueError("Habitat and resistance rasters have different transforms.")

        if hab_src.width != res_src.width or hab_src.height != res_src.height:
            raise ValueError("Habitat and resistance rasters have different shapes.")

    @property
    def n_planning_units(self) -> int:
        return int(len(self.planning_units))

    @property
    def planning_unit_ids(self) -> list[int]:
        return self.planning_units[self.internal_id_column].astype(int).tolist()

    @property
    def default_budget(self) -> int:
        """
        Version 1 default: one unit of budget per polygon.
        """
        return self.n_planning_units

    def get_planning_unit_row(self, pu_id: int) -> pd.Series:
        matches = self.planning_units.loc[
            self.planning_units[self.internal_id_column] == pu_id
        ]
        if matches.empty:
            raise KeyError(f"Planning unit with pu_id={pu_id} not found.")
        return matches.iloc[0]

    def get_cost(self, pu_id: int) -> float:
        row = self.get_planning_unit_row(pu_id)
        return float(row[self.cost_column])

    def selected_geodataframe(self, pu_ids: list[int]) -> gpd.GeoDataFrame:
        if not pu_ids:
            return self.planning_units.iloc[0:0].copy()
        return self.planning_units.loc[
            self.planning_units[self.internal_id_column].isin(pu_ids)
        ].copy()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "vector_path": str(self.vector_path),
            "habitat_raster_path": str(self.habitat_raster_path),
            "resistance_raster_path": str(self.resistance_raster_path),
            "n_planning_units": self.n_planning_units,
            "id_column": self.id_column,
            "internal_id_column": self.internal_id_column,
            "cost_column": self.cost_column,
            "eligibility_column": self.eligibility_column,
            "restored_resistance_value": self.restored_resistance_value,
            "raster_width": self.raster_width,
            "raster_height": self.raster_height,
            "raster_crs": self.raster_crs,
        }
