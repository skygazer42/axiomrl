from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from collections.abc import Sequence

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.manager import ExperimentManager
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.trainer import TrainResult, Trainer
from rl_training.runtime.workflows import resume_training


@dataclass(slots=True)
class FunctionTrainer(Trainer):
    _train_fn: Callable[[], TrainResult]

    def train(self) -> TrainResult:
        return self._train_fn()


class DefaultExperimentManager(ExperimentManager):
    def setup(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Trainer:
        spec = get_algorithm_spec(config.algo)
        return FunctionTrainer(_train_fn=lambda: spec.train_fn(config, callbacks=callbacks))

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        total_timesteps: int | None = None,
        output_dir: str | Path | None = None,
        eval_episodes: int | None = None,
        run_suffix: str | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> Trainer:
        return FunctionTrainer(
            _train_fn=lambda: resume_training(
                checkpoint_path,
                total_timesteps=total_timesteps,
                output_dir=output_dir,
                eval_episodes=eval_episodes,
                run_suffix=run_suffix,
                callbacks=callbacks,
            )
        )
