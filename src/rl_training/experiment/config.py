from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    algo: str
    env_id: str
    seed: int
    total_timesteps: int
    output_dir: Path
    execution_backend: str = "local_sync"
    device: str = "auto"
    num_envs: int = 1
    eval_episodes: int = 5
    log_interval: int = 1
    checkpoint_interval: int = 1
    tags: tuple[str, ...] = ()
    benchmark: dict[str, Any] = field(default_factory=dict)
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
