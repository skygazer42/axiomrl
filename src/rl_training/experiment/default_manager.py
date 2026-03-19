from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.manager import ExperimentManager
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.runner import FunctionRunner, Runner
from rl_training.runtime.trainer import TrainResult, Trainer
from rl_training.runtime.workflows import resume_training


@dataclass(slots=True)
class FunctionTrainer(Trainer):
    runner: Runner

    def train(self) -> TrainResult:
        return self.runner.run()


class DefaultExperimentManager(ExperimentManager):
    def setup_runner(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Runner:
        spec = get_algorithm_spec(config.algo)
        return FunctionRunner(run_fn=lambda: spec.train_fn(config, callbacks=callbacks))

    def setup(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Trainer:
        return FunctionTrainer(runner=self.setup_runner(config, callbacks=callbacks))

    def resume_runner(
        self,
        checkpoint_path: str | Path,
        *,
        total_timesteps: int | None = None,
        output_dir: str | Path | None = None,
        eval_episodes: int | None = None,
        run_suffix: str | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> Runner:
        return FunctionRunner(
            run_fn=lambda: resume_training(
                checkpoint_path,
                total_timesteps=total_timesteps,
                output_dir=output_dir,
                eval_episodes=eval_episodes,
                run_suffix=run_suffix,
                callbacks=callbacks,
            )
        )

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
            runner=self.resume_runner(
                checkpoint_path,
                total_timesteps=total_timesteps,
                output_dir=output_dir,
                eval_episodes=eval_episodes,
                run_suffix=run_suffix,
                callbacks=callbacks,
            )
        )
