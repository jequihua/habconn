"""SB3 callbacks for lightweight artifact-first logging.

Intentionally small. The one useful callback here is a JSONL episode
logger that writes one record per completed training episode — enough
to produce a simple training-history artifact without pulling in
tensorboard or other heavy tooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeHistoryCallback(BaseCallback):
    """Append one JSONL record per completed training episode.

    Each record contains:
        step            : cumulative training timestep
        episode_reward  : sum of rewards in the episode
        episode_length  : number of steps in the episode
        final_pc        : last PC value recorded in info (if present)

    The Monitor wrapper (added automatically by SB3 when an env is
    wrapped in DummyVecEnv / single-env) makes episode rewards available
    in ``info["episode"]``. We read that here.
    """

    def __init__(self, output_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any previous file so each run starts fresh.
        self._output_path.write_text("", encoding="utf-8")
        self._episode_count = 0

    def _on_step(self) -> bool:
        # self.locals["infos"] is a list (per parallel env) of dicts.
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is None:
                continue
            record = {
                "step": int(self.num_timesteps),
                "episode": self._episode_count,
                "episode_reward": float(ep.get("r", 0.0)),
                "episode_length": int(ep.get("l", 0)),
                "final_pc": float(info.get("pc_value", 0.0)) if "pc_value" in info else None,
            }
            with self._output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._episode_count += 1
        return True

    @property
    def episode_count(self) -> int:
        return self._episode_count
