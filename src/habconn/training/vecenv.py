"""Worker-safe vectorized environment construction.

Builds a ``DummyVecEnv`` of ``VectorHabitatEnv`` instances where each
worker has its own Graphab work root and a deterministic per-worker
seed derived from the base seed. Each sub-env is wrapped in
``stable_baselines3.common.monitor.Monitor`` so SB3's
``info["episode"]`` machinery (and the existing ``EpisodeHistoryCallback``)
keeps working.

Scope notes:
- ``DummyVecEnv`` runs sub-environments serially in the same process.
  This is the safest first vectorized path on Windows and with a
  Graphab CLI subprocess: it avoids fork/spawn complications and
  multi-process JVM startup overhead. ``SubprocVecEnv`` is deferred.
- Each worker gets ``<base_work_root>/worker_NNN/`` so Graphab scratch
  directories cannot collide.
- Per-worker seeds are derived deterministically from the base seed
  via ``worker_seed(base_seed, worker_index)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from habconn.problems.candidate_generation import CandidateRankingStrategy
from habconn.training.make_env import make_env


def worker_work_root(base_work_root: Path, worker_index: int) -> Path:
    """Return the isolated Graphab work root for ``worker_index``.

    The pattern is ``<base_work_root>/worker_NNN/`` with a zero-padded
    three-digit index so workers stay sortable.
    """
    if worker_index < 0:
        raise ValueError(f"worker_index must be >= 0, got {worker_index}")
    return Path(base_work_root) / f"worker_{worker_index:03d}"


def worker_seed(base_seed: Optional[int], worker_index: int) -> Optional[int]:
    """Derive a deterministic per-worker seed from a base seed.

    Returns ``None`` when ``base_seed`` is ``None`` (callers may pass a
    seedless config). Otherwise returns ``base_seed + worker_index``,
    masked to a non-negative 32-bit value so it is acceptable to
    downstream RNG seeding.
    """
    if base_seed is None:
        return None
    return int((int(base_seed) + int(worker_index)) & 0x7FFFFFFF)


def worker_work_roots(base_work_root: Path, n_envs: int) -> List[Path]:
    """Return the list of per-worker work roots for ``n_envs`` workers."""
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    return [worker_work_root(base_work_root, i) for i in range(n_envs)]


def make_vector_envs(
    *,
    n_envs: int,
    data_dir: Path,
    graphab_jar: Path,
    work_root: Path,
    budget: int = 3,
    k: int = 10,
    java_executable: str = "java",
    jvm_memory: str = "4G",
    ranking_strategy: CandidateRankingStrategy = CandidateRankingStrategy.BY_PU_ID,
    base_seed: Optional[int] = None,
    enable_cache: bool = True,
) -> DummyVecEnv:
    """Build a ``DummyVecEnv`` of ``n_envs`` independent ``VectorHabitatEnv`` workers.

    Parameters
    ----------
    n_envs : int
        Number of parallel sub-environments. Must be ``>= 1``.
    work_root : Path
        Base Graphab scratch directory. Each worker gets its own
        ``<work_root>/worker_NNN/`` underneath.
    base_seed : int or None
        Base seed for deterministic per-worker seeding. ``None`` means
        unseeded.

    All other parameters mirror :func:`habconn.training.make_env.make_env`.

    Returns
    -------
    DummyVecEnv
        Vectorized environment ready to pass to MaskablePPO. Each
        sub-env exposes ``action_masks()`` and is wrapped in
        ``Monitor`` so SB3's ``info["episode"]`` machinery is enabled.

    Notes
    -----
    Worker factories are callables passed to ``DummyVecEnv``;
    ``DummyVecEnv`` invokes them during vector-env initialization, so
    each sub-environment is constructed by the time this function
    returns. The closures returned by ``_make_factory`` capture the
    per-worker scratch directory and seed so the factory itself stays
    a small lambda.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    base_root = Path(work_root)

    def _make_factory(worker_index: int) -> Callable[[], object]:
        sub_root = worker_work_root(base_root, worker_index)
        sub_seed = worker_seed(base_seed, worker_index)

        def _factory():
            env = make_env(
                data_dir=data_dir,
                graphab_jar=graphab_jar,
                work_root=sub_root,
                budget=budget,
                k=k,
                java_executable=java_executable,
                jvm_memory=jvm_memory,
                ranking_strategy=ranking_strategy,
                random_seed=sub_seed,
                enable_cache=enable_cache,
            )
            return Monitor(env)

        return _factory

    return DummyVecEnv([_make_factory(i) for i in range(n_envs)])
