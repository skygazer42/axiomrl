from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rl_training.runtime.trainer import TrainResult


class Runner(Protocol):
    def run(self) -> TrainResult:
        ...


@dataclass(slots=True)
class FunctionRunner(Runner):
    run_fn: Callable[[], TrainResult]

    def run(self) -> TrainResult:
        return self.run_fn()
