from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.runner import Runner
from axiomrl.runtime.trainer import Trainer


class ExperimentManager(Protocol):
    def setup(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Trainer: ...

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        total_timesteps: int | None = None,
        output_dir: str | Path | None = None,
        eval_episodes: int | None = None,
        run_suffix: str | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> Trainer: ...

    def setup_runner(self, config: TrainConfig, *, callbacks: Sequence[Callback] | None = None) -> Runner: ...

    def resume_runner(
        self,
        checkpoint_path: str | Path,
        *,
        total_timesteps: int | None = None,
        output_dir: str | Path | None = None,
        eval_episodes: int | None = None,
        run_suffix: str | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> Runner: ...
