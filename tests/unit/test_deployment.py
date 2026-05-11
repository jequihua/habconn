"""Unit tests for the deployment-export helper.

These tests exercise ``training/deployment.py`` without Graphab and
without a real SB3 model. ``MaskablePPO.load`` is monkeypatched to
return a small stub model that records the masks it was given; the
env is a hand-rolled gymnasium.Env that mirrors the surface
``run_deployment_export`` consumes (reset / step / action_masks);
the problem is a thin object exposing the public attributes the
helper reads (``internal_id_column``, ``id_column``, ``cost_column``,
``selected_geodataframe``).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import gymnasium as gym
import numpy as np
import pytest
from shapely.geometry import Point

from habconn.training import deployment as deployment_mod
from habconn.training.deployment import (
    CSV_FILENAME,
    GEOPACKAGE_FILENAME,
    SUMMARY_FILENAME,
    run_deployment_export,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEnv(gym.Env):
    """Multi-step env that emits a fixed action sequence.

    The env appends the chosen action to ``self._selected`` and exposes
    that list via ``info['selected_pu_ids']`` so the deployment helper
    can read selection order from the final info dict, matching the
    real ``VectorHabitatEnv``.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, k: int = 4, steps: int = 3, baseline_pc: float = 0.0001) -> None:
        super().__init__()
        self.k = k
        self.steps = steps
        self.baseline_pc = baseline_pc
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(k,), dtype=np.bool_),
            "candidate_ids": gym.spaces.Box(-1, 2**31 - 1, shape=(k,), dtype=np.int32),
            "candidate_costs": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
            "candidate_areas": gym.spaces.Box(0, 1e6, shape=(k,), dtype=np.float32),
            "remaining_budget": gym.spaces.Box(0, 1e6, shape=(1,), dtype=np.float32),
            "current_pc": gym.spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(k)
        self._step_idx = 0
        self._selected: list[int] = []
        self.mask_history: list[np.ndarray] = []
        self._current_pc = baseline_pc

    def _obs(self) -> dict:
        mask = np.array(
            [i < self.steps - self._step_idx for i in range(self.k)],
            dtype=np.bool_,
        )
        # If we've already exhausted the budget, no valid candidates.
        if self._step_idx >= self.steps:
            mask = np.zeros((self.k,), dtype=np.bool_)
        return {
            "action_mask": mask,
            "candidate_ids": np.arange(self.k, dtype=np.int32),
            "candidate_costs": np.ones((self.k,), dtype=np.float32),
            "candidate_areas": np.full((self.k,), 10.0, dtype=np.float32),
            "remaining_budget": np.array(
                [float(self.steps - self._step_idx)], dtype=np.float32,
            ),
            "current_pc": np.array([self._current_pc], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        self._step_idx = 0
        self._selected = []
        self.mask_history = []
        self._current_pc = self.baseline_pc
        return self._obs(), {"pc_value": self.baseline_pc, "selected_pu_ids": []}

    def action_masks(self) -> np.ndarray:
        if self._step_idx >= self.steps:
            return np.zeros((self.k,), dtype=bool)
        return np.ones((self.k,), dtype=bool)

    def step(self, action: int):
        self._step_idx += 1
        self._selected.append(int(action))
        self._current_pc = self.baseline_pc + 1e-5 * self._step_idx
        terminated = self._step_idx >= self.steps
        info = {"pc_value": self._current_pc, "selected_pu_ids": list(self._selected)}
        return self._obs(), 1e-5, terminated, False, info


class _StubModel:
    """Returns a pre-baked action sequence and records mask shapes."""

    def __init__(self, actions: list[int]) -> None:
        self.actions = list(actions)
        self._i = 0
        self.received_masks: list[np.ndarray] = []
        self.predict_calls = 0

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if action_masks is None:
            raise AssertionError("Deployment must pass action_masks to model.predict")
        self.received_masks.append(np.asarray(action_masks))
        self.predict_calls += 1
        a = self.actions[self._i]
        self._i += 1
        return a, None


class _StubProblem:
    """Public surface the deployment helper consumes."""

    internal_id_column = "pu_id"
    id_column = "lyr_1"
    cost_column = "cost"

    def __init__(self, *, pu_ids: list[int]) -> None:
        # All planning units this problem knows about.
        self._gdf = gpd.GeoDataFrame(
            {
                self.internal_id_column: pu_ids,
                self.id_column: [f"src_{p}" for p in pu_ids],
                self.cost_column: [1.0 for _ in pu_ids],
                "geometry": [Point(float(p), 0.0) for p in pu_ids],
            },
            crs="EPSG:4326",
        )

    def selected_geodataframe(self, pu_ids: list[int]) -> gpd.GeoDataFrame:
        if not pu_ids:
            return self._gdf.iloc[0:0].copy()
        return self._gdf.loc[self._gdf[self.internal_id_column].isin(pu_ids)].copy()


def _install_stub_model(monkeypatch, actions: list[int]) -> _StubModel:
    stub = _StubModel(actions)

    class _StubMaskablePPO:
        @staticmethod
        def load(path):
            assert Path(path).exists(), f"Model path must exist: {path}"
            return stub

    # ``run_deployment_export`` imports MaskablePPO lazily. Patch it on
    # ``sb3_contrib`` so the local lazy import inside the helper
    # resolves to our stub.
    import sb3_contrib
    monkeypatch.setattr(sb3_contrib, "MaskablePPO", _StubMaskablePPO)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_run(tmp_path: Path):
    model_path = tmp_path / "best_model.zip"
    model_path.write_bytes(b"")  # presence-only; loader is stubbed
    return tmp_path / "deployment", model_path


class TestRunDeploymentExport:
    def test_summary_records_selected_pus_in_order(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        stub_model = _install_stub_model(monkeypatch, actions=[2, 0, 1])
        env = _StubEnv(k=4, steps=3)
        problem = _StubProblem(pu_ids=[0, 1, 2, 3])

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
            run_name="unit_run",
            budget=3,
            k=4,
        )

        # Selection order is the order of env.step actions.
        assert summary["selected_pu_ids"] == [2, 0, 1]
        assert summary["n_selected"] == 3
        assert summary["episode_steps"] == 3
        assert summary["deterministic"] is True
        # Predict was called once per env step, with action_masks each time.
        assert stub_model.predict_calls == 3
        for m in stub_model.received_masks:
            assert m.dtype == bool

    def test_on_disk_summary_is_json_serializable(self, tmp_path, monkeypatch):
        """The on-disk ``deployment_summary.json`` is the authoritative
        JSON contract. The in-memory return dict carries an extra
        underscore-prefixed inspection side channel (numpy arrays); it
        is intentionally not serialized to disk.
        """
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[0, 1])
        env = _StubEnv(k=3, steps=2)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
        )

        # Read the on-disk file and round-trip it through json.
        on_disk_text = (out_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
        restored = json.loads(on_disk_text)
        assert restored["selected_pu_ids"] == [0, 1]
        # No underscore-prefixed inspection side channel ever lands on disk.
        for key in restored:
            assert not key.startswith("_"), f"underscore key leaked into JSON: {key}"

        # The in-memory dict, by contrast, carries the inspection side
        # channel for the trainer / inspection writer to consume.
        assert "_initial_observation" in summary
        assert "_trace_steps" in summary

    def test_csv_columns_include_selection_order(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[2, 0])
        env = _StubEnv(k=3, steps=2)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
        )

        csv_path = out_dir / CSV_FILENAME
        assert csv_path.exists()
        rows = csv_path.read_text(encoding="utf-8").splitlines()
        header = rows[0].split(",")
        assert "selection_order" in header
        assert problem.internal_id_column in header

        # Rows are in selection order (2 first, then 0).
        data = [r.split(",") for r in rows[1:]]
        pu_idx = header.index(problem.internal_id_column)
        ord_idx = header.index("selection_order")
        order_by_pu = {int(r[pu_idx]): int(r[ord_idx]) for r in data}
        assert order_by_pu == {2: 0, 0: 1}

    def test_empty_selected_writes_valid_empty_artifacts(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[])
        # Zero-step env: terminates immediately on reset's next mask check.
        env = _StubEnv(k=3, steps=0)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
        )

        assert summary["selected_pu_ids"] == []
        assert summary["n_selected"] == 0
        assert summary["episode_steps"] == 0

        # All three contract artifacts still exist.
        assert (out_dir / SUMMARY_FILENAME).exists()
        gpkg = out_dir / GEOPACKAGE_FILENAME
        csv = out_dir / CSV_FILENAME
        assert gpkg.exists()
        assert csv.exists()
        # CSV still has a header row.
        assert csv.read_text(encoding="utf-8").splitlines()[0].count(",") >= 1

    def test_model_path_recorded(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[0])
        env = _StubEnv(k=2, steps=1)
        problem = _StubProblem(pu_ids=[0, 1])

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
            source_model_selection_path=tmp_path / "selection" / "model_selection.json",
        )

        assert summary["model_path"] == str(model_path)
        assert summary["model_selection_path"] == str(
            tmp_path / "selection" / "model_selection.json"
        )
        assert summary["deployment_summary_path"] == str(out_dir / SUMMARY_FILENAME)

    def test_gpkg_round_trips_for_nonempty_selection(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[2, 0])
        env = _StubEnv(k=3, steps=2)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
        )

        gpkg = out_dir / GEOPACKAGE_FILENAME
        assert gpkg.exists() and gpkg.stat().st_size > 0
        try:
            loaded = gpd.read_file(gpkg)
        except Exception as e:  # pragma: no cover - environment-dependent
            pytest.skip(f"GeoPackage reader not available: {e}")
        assert set(loaded[problem.internal_id_column].astype(int)) == set(
            summary["selected_pu_ids"]
        )


class TestWriteGeopackageFailureSemantics:
    """Empty selection → placeholder; non-empty selection writer failure
    must raise instead of silently producing a misleading placeholder.
    """

    def test_nonempty_writer_failure_propagates(self, tmp_path, monkeypatch):
        from habconn.training import deployment as deployment_mod

        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[1, 0])
        env = _StubEnv(k=3, steps=2)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        class _BoomError(RuntimeError):
            pass

        def _boom(self, *args, **kwargs):
            raise _BoomError("simulated GPKG writer failure")

        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _boom)

        with pytest.raises(_BoomError):
            run_deployment_export(
                model_path=model_path,
                env=env,
                problem=problem,
                output_dir=out_dir,
            )

    def test_empty_selection_still_falls_back_to_placeholder(self, tmp_path, monkeypatch):
        out_dir, model_path = _make_run(tmp_path)
        _install_stub_model(monkeypatch, actions=[])
        env = _StubEnv(k=3, steps=0)
        problem = _StubProblem(pu_ids=[0, 1, 2])

        # Patch to_file to a sentinel that would raise if reached.
        called = {"hit": False}

        def _should_not_be_called(self, *args, **kwargs):
            called["hit"] = True
            raise AssertionError("to_file must not be called for empty selection")

        monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _should_not_be_called)

        summary = run_deployment_export(
            model_path=model_path,
            env=env,
            problem=problem,
            output_dir=out_dir,
        )

        assert summary["n_selected"] == 0
        assert (out_dir / GEOPACKAGE_FILENAME).exists()
        assert called["hit"] is False


class TestExperimentPathsHasDeploymentFields:
    """The contract layout must advertise the deployment paths."""

    def test_layout(self, tmp_path):
        from habconn.training.experiment import ExperimentConfig, ExperimentPaths
        cfg = ExperimentConfig(
            run_name="ok",
            data_dir=tmp_path / "d",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "out",
        )
        paths = ExperimentPaths.from_config(cfg)
        run_dir = tmp_path / "out" / "ok"
        assert paths.deployment_dir == run_dir / "deployment"
        assert paths.deployment_summary_path == run_dir / "deployment" / SUMMARY_FILENAME
        assert paths.selected_planning_units_gpkg_path == run_dir / "deployment" / GEOPACKAGE_FILENAME
        assert paths.selected_planning_units_csv_path == run_dir / "deployment" / CSV_FILENAME

    def test_ensure_creates_deployment_dir(self, tmp_path):
        from habconn.training.experiment import ExperimentConfig, ExperimentPaths
        cfg = ExperimentConfig(
            run_name="ok",
            data_dir=tmp_path / "d",
            graphab_jar=tmp_path / "g.jar",
            work_root=tmp_path / "w",
            output_root=tmp_path / "out",
        )
        paths = ExperimentPaths.from_config(cfg)
        paths.ensure()
        assert paths.deployment_dir.is_dir()
