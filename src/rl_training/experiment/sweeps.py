from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_training.experiment.config import TrainConfig, resolve_benchmark_seed_values
from rl_training.runtime.types import MetricDict


@dataclass(frozen=True, slots=True)
class SeedSweepPlan:
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkRunRecord:
    seed: int
    run_dir: Path
    checkpoint_path: Path | None
    metrics: MetricDict


def resolve_benchmark_seeds(config: TrainConfig) -> tuple[int, ...]:
    resolved = resolve_benchmark_seed_values(config.benchmark)
    if resolved is None:
        return ()
    return resolved
