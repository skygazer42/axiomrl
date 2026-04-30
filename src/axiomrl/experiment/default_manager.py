from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.manager import ExperimentManager
from axiomrl.experiment.registry import get_algorithm_spec
from axiomrl.experiment.sweeps import SeedSweepPlan, resolve_benchmark_seeds
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.runner import BenchmarkRunner, FunctionRunner, Runner
from axiomrl.runtime.trainer import Trainer, TrainResult
from axiomrl.runtime.workflows import resume_training


@dataclass(slots=True)
class FunctionTrainer(Trainer):
    runner: Runner

    def train(self) -> TrainResult:
        return self.runner.run()


class DefaultExperimentManager(ExperimentManager):
    def setup_runner(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Runner:
        sweep_seeds = resolve_benchmark_seeds(config)
        spec = get_algorithm_spec(config.algo)
        if sweep_seeds:
            summary_path = config.output_dir / "benchmark-summary.json"
            seed_sweep = SeedSweepPlan(seeds=sweep_seeds)

            def _make_runner(seed: int) -> Runner:
                child_benchmark = dict(config.benchmark)
                child_benchmark.pop("seeds", None)
                child_config = replace(config, seed=seed, benchmark=child_benchmark)
                seed_callbacks = None if callbacks is None else tuple(deepcopy(tuple(callbacks)))
                return FunctionRunner(run_fn=lambda: spec.train_fn(child_config, callbacks=seed_callbacks))

            return BenchmarkRunner(
                seed_sweep=seed_sweep,
                make_runner=_make_runner,
                summary_path=summary_path,
            )

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
