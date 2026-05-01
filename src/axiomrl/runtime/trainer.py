from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axiomrl.runtime.types import MetricDict


@dataclass(slots=True)
class TrainResult:
    run_dir: Path
    checkpoint_path: Path | None
    metrics: MetricDict
    benchmark_summary_path: Path | None = None


@dataclass(slots=True)
class TrainerState:
    algorithm: str
    run_dir: Path
    global_step: int = 0
    epoch: int = 0
    update_count: int = 0
    should_stop: bool = False
    stop_reason: str | None = None

    def request_stop(self, reason: str) -> None:
        self.should_stop = True
        if self.stop_reason is None:
            self.stop_reason = reason


class Trainer(Protocol):
    def train(self) -> TrainResult: ...
