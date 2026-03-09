from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rl_training.algorithms.base import UpdateResult
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


class Callback(Protocol):
    def on_train_start(self, trainer: object) -> None:
        ...

    def on_collect_end(self, trainer: object, result: CollectResult) -> None:
        ...

    def on_update_end(self, trainer: object, result: UpdateResult) -> None:
        ...

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None:
        ...

    def on_train_end(self, trainer: object, result: TrainResult) -> None:
        ...


class CallbackList:
    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks = tuple(callbacks or ())

    def on_train_start(self, trainer: object) -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_collect_end(self, trainer: object, result: CollectResult) -> None:
        for callback in self.callbacks:
            callback.on_collect_end(trainer, result)

    def on_update_end(self, trainer: object, result: UpdateResult) -> None:
        for callback in self.callbacks:
            callback.on_update_end(trainer, result)

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None:
        for callback in self.callbacks:
            callback.on_eval_end(trainer, metrics)

    def on_train_end(self, trainer: object, result: TrainResult) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer, result)
