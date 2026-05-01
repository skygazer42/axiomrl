from collections.abc import Sequence
from typing import Protocol

from axiomrl.algorithms.base import UpdateResult
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.trainer import TrainResult
from axiomrl.runtime.types import MetricDict


class Callback(Protocol):
    def on_train_start(self, trainer: object) -> None: ...

    def on_collect_end(self, trainer: object, result: CollectResult) -> None: ...

    def on_update_end(self, trainer: object, result: UpdateResult) -> None: ...

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None: ...

    def on_train_end(self, trainer: object, result: TrainResult) -> None: ...


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


def merge_callbacks(*callback_groups: Sequence[Callback] | None) -> tuple[Callback, ...]:
    merged: list[Callback] = []
    for callback_group in callback_groups:
        if callback_group is None:
            continue
        merged.extend(callback_group)
    return tuple(merged)
