from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.trainer import Trainer


class ExperimentManager(Protocol):
    def setup(self, config: TrainConfig) -> Trainer:
        ...

    def resume(self, checkpoint_path: str | Path) -> Trainer:
        ...
