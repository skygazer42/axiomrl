from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rl_training.runtime.types import MetricDict


@dataclass(slots=True)
class TrainResult:
    run_dir: Path
    checkpoint_path: Path | None
    metrics: MetricDict


@dataclass(slots=True)
class TrainerState:
    algorithm: str
    run_dir: Path
    global_step: int = 0


class Trainer(Protocol):
    def train(self) -> TrainResult:
        ...
