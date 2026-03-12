from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rl_training.algorithms.base import UpdateResult
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    metric: str = "eval_return_mean"
    mode: str = "max"
    patience: int = 5
    min_delta: float = 0.0
    min_steps: int = 0
    target_value: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {self.min_steps}")


class EarlyStoppingCallback:
    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_value: float | None = None
        self.bad_eval_count = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> EarlyStoppingCallback:
        return cls(
            EarlyStoppingConfig(
                metric=str(payload.get("metric", "eval_return_mean")),
                mode=str(payload.get("mode", "max")),
                patience=int(payload.get("patience", 5)),
                min_delta=float(payload.get("min_delta", 0.0)),
                min_steps=int(payload.get("min_steps", 0)),
                target_value=float(payload["target_value"]) if payload.get("target_value") is not None else None,
            )
        )

    def on_train_start(self, trainer: object) -> None:
        del trainer

    def on_collect_end(self, trainer: object, result: CollectResult) -> None:
        del trainer, result

    def on_update_end(self, trainer: object, result: UpdateResult) -> None:
        del trainer, result

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None:
        if not isinstance(trainer, TrainerState):
            return
        if trainer.global_step < self.config.min_steps:
            return

        metric_value = metrics.get(self.config.metric)
        if metric_value is None:
            return

        current = float(metric_value)
        if self.config.target_value is not None:
            reached_target = current >= self.config.target_value if self.config.mode == "max" else current <= self.config.target_value
            if reached_target:
                trainer.request_stop(
                    f"early stopping target reached: {self.config.metric}={current:.6f} target={self.config.target_value:.6f}"
                )
                return

        if self.best_value is None:
            self.best_value = current
            self.bad_eval_count = 0
            return

        if self._is_improvement(current):
            self.best_value = current
            self.bad_eval_count = 0
            return

        self.bad_eval_count += 1
        if self.bad_eval_count > self.config.patience:
            trainer.request_stop(
                f"early stopping patience exhausted on {self.config.metric}: current={current:.6f} best={self.best_value:.6f}"
            )

    def on_train_end(self, trainer: object, result: TrainResult) -> None:
        del trainer, result

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.config.mode == "max":
            return current > self.best_value + self.config.min_delta
        return current < self.best_value - self.config.min_delta


def build_control_callbacks(config: TrainConfig) -> tuple[Callback, ...]:
    early_stopping = config.algo_kwargs.get("early_stopping")
    if early_stopping in (None, False):
        return ()
    if not isinstance(early_stopping, Mapping):
        raise TypeError(f"expected algo_kwargs['early_stopping'] to be a mapping, got {type(early_stopping)!r}")
    return (EarlyStoppingCallback.from_mapping(early_stopping),)


def resolve_eval_interval(config: TrainConfig) -> int:
    eval_interval = int(config.algo_kwargs.get("eval_interval", config.total_timesteps))
    if eval_interval < 1:
        raise ValueError(f"eval_interval must be >= 1, got {eval_interval}")
    return eval_interval


def resolve_max_updates(config: TrainConfig) -> int | None:
    value = config.algo_kwargs.get("max_updates")
    if value in (None, False):
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"max_updates must be >= 1, got {resolved}")
    return resolved


def resolve_max_epochs(config: TrainConfig) -> int | None:
    value = config.algo_kwargs.get("max_epochs")
    if value in (None, False):
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"max_epochs must be >= 1, got {resolved}")
    return resolved


def resolve_min_buffer_size(config: TrainConfig, *, default: int = 0) -> int:
    value = config.algo_kwargs.get("min_buffer_size", default)
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"min_buffer_size must be >= 0, got {resolved}")
    return resolved


def resolve_effective_total_updates(config: TrainConfig) -> int:
    candidates = [int(config.total_timesteps)]
    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    if max_updates is not None:
        candidates.append(max_updates)
    if max_epochs is not None:
        candidates.append(max_epochs)
    resolved = min(candidates)
    if resolved < 1:
        raise ValueError(f"effective total updates must be >= 1, got {resolved}")
    return resolved


def stop_reason_for_training_limits(
    *,
    epoch: int,
    update_count: int,
    max_epochs: int | None,
    max_updates: int | None,
) -> str | None:
    if max_epochs is not None and epoch >= max_epochs:
        return f"max_epochs reached: epoch={epoch} limit={max_epochs}"
    if max_updates is not None and update_count >= max_updates:
        return f"max_updates reached: update_count={update_count} limit={max_updates}"
    return None


def should_run_periodic_eval(*, global_step: int, total_timesteps: int, eval_interval: int) -> bool:
    return global_step % eval_interval == 0 or global_step >= total_timesteps


def should_run_evaluation(*, global_step: int, total_timesteps: int, eval_interval: int) -> bool:
    return should_run_periodic_eval(
        global_step=global_step,
        total_timesteps=total_timesteps,
        eval_interval=eval_interval,
    )
