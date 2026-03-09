from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_training.runtime.types import MetricDict


@dataclass(slots=True)
class EvalResult:
    num_episodes: int
    metrics: MetricDict


class Evaluator(Protocol):
    def evaluate(self, *, num_episodes: int) -> EvalResult:
        ...
