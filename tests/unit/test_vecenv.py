"""Unit tests for the vectorized-env helpers.

These tests exercise the parts of ``training/vecenv.py`` that do not
require Graphab: per-worker work-root layout, deterministic seed
derivation, and the n_envs guard on ``make_vector_envs``. The
full-factory closure path is also covered: ``DummyVecEnv`` invokes
worker factories during its initialization, so we patch ``make_env``
to capture each invocation and assert that every worker received an
isolated work root and a deterministic per-worker seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from habconn.training.vecenv import (
    make_vector_envs,
    worker_seed,
    worker_work_root,
    worker_work_roots,
)


class TestWorkerWorkRoot:
    def test_layout(self, tmp_path):
        root = worker_work_root(tmp_path, 0)
        assert root == tmp_path / "worker_000"

    def test_pads_three_digits(self, tmp_path):
        assert worker_work_root(tmp_path, 7) == tmp_path / "worker_007"
        assert worker_work_root(tmp_path, 42) == tmp_path / "worker_042"
        assert worker_work_root(tmp_path, 123) == tmp_path / "worker_123"

    def test_negative_index_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            worker_work_root(tmp_path, -1)


class TestWorkerWorkRoots:
    def test_isolated_per_worker(self, tmp_path):
        roots = worker_work_roots(tmp_path, 4)
        assert len(roots) == 4
        # All distinct, all under the base, all matching the pattern.
        assert len(set(roots)) == 4
        for i, r in enumerate(roots):
            assert r == tmp_path / f"worker_{i:03d}"
            assert r.parent == tmp_path

    def test_n_envs_zero_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            worker_work_roots(tmp_path, 0)

    def test_n_envs_negative_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            worker_work_roots(tmp_path, -2)


class TestWorkerSeed:
    def test_deterministic(self):
        assert worker_seed(42, 0) == worker_seed(42, 0)
        assert worker_seed(42, 1) == worker_seed(42, 1)

    def test_distinct_across_workers(self):
        seeds = [worker_seed(42, i) for i in range(8)]
        assert len(set(seeds)) == 8

    def test_none_base_returns_none(self):
        assert worker_seed(None, 0) is None
        assert worker_seed(None, 5) is None

    def test_non_negative_32bit(self):
        # Seeds passed to numpy/PPO must fit in a non-negative 32-bit int.
        for base in (0, 1, 42, 1_000_000):
            for idx in (0, 1, 7, 1024):
                s = worker_seed(base, idx)
                assert isinstance(s, int)
                assert 0 <= s <= 0x7FFFFFFF


class TestMakeVectorEnvsGuards:
    """Cover the n_envs guards and verify per-worker isolation in the
    factories that ``DummyVecEnv`` invokes during initialization."""

    def test_n_envs_zero_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            make_vector_envs(
                n_envs=0,
                data_dir=tmp_path / "data",
                graphab_jar=tmp_path / "graphab.jar",
                work_root=tmp_path / "scratch",
                base_seed=42,
            )

    def test_n_envs_negative_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            make_vector_envs(
                n_envs=-1,
                data_dir=tmp_path / "data",
                graphab_jar=tmp_path / "graphab.jar",
                work_root=tmp_path / "scratch",
                base_seed=42,
            )

    def test_factories_carry_isolated_roots_and_seeds(self, tmp_path):
        """``DummyVecEnv`` invokes worker factories during initialization,
        so we patch ``make_env`` to capture each call and assert that
        every worker received an isolated work root and a deterministic
        per-worker seed.
        """
        from habconn.training import vecenv as vecenv_mod

        captured: list[dict] = []

        class _StubEnv:
            # Minimal stub that satisfies what DummyVecEnv + Monitor need at
            # construction time: a Gymnasium-compatible env with action_space
            # and observation_space attributes is required, so we install a
            # Monitor-friendly fake by subclassing gym.Env.
            pass

        import gymnasium as gym
        import numpy as np

        class _FakeEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
                self.action_space = gym.spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                return np.zeros((1,), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((1,), dtype=np.float32), 0.0, True, False, {}

            def action_masks(self):
                return np.array([True, False], dtype=bool)

        def _fake_make_env(**kwargs):
            captured.append(kwargs)
            return _FakeEnv()

        monkey = vecenv_mod.make_env
        try:
            vecenv_mod.make_env = _fake_make_env

            vec = make_vector_envs(
                n_envs=3,
                data_dir=tmp_path / "data",
                graphab_jar=tmp_path / "graphab.jar",
                work_root=tmp_path / "scratch",
                budget=2,
                k=4,
                base_seed=100,
            )
        finally:
            vecenv_mod.make_env = monkey

        # Three sub-envs constructed.
        assert vec.num_envs == 3
        assert len(captured) == 3

        # Each call got an isolated work root.
        roots = [Path(c["work_root"]) for c in captured]
        assert roots[0] == tmp_path / "scratch" / "worker_000"
        assert roots[1] == tmp_path / "scratch" / "worker_001"
        assert roots[2] == tmp_path / "scratch" / "worker_002"
        assert len(set(roots)) == 3

        # Each call got a distinct deterministic seed derived from base_seed.
        seeds = [c["random_seed"] for c in captured]
        assert seeds == [100, 101, 102]

        # Action masks dispatch correctly through DummyVecEnv + Monitor.
        masks = vec.env_method("action_masks")
        assert len(masks) == 3
        for m in masks:
            assert m.dtype == bool
            assert m.shape == (2,)
            assert m.tolist() == [True, False]
