from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def resolve_benchmark_seed_values(benchmark: Mapping[str, object]) -> tuple[int, ...] | None:
    """Normalize benchmark sweep seeds into a distinct, ordered tuple of non-negative ints."""
    requested = benchmark.get("seeds")
    if requested is None:
        return None
    if isinstance(requested, str | bytes) or not isinstance(requested, Sequence):
        raise TypeError("benchmark['seeds'] must be a sequence of integers")
    if len(requested) == 0:
        raise ValueError("benchmark['seeds'] must not be empty")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("benchmark['seeds'] entries must be integers")
        if value < 0:
            raise ValueError("benchmark['seeds'] entries must be non-negative integers")
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)
