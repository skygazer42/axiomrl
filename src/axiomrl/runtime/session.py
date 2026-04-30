from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.callbacks import Callback, CallbackList, merge_callbacks
from axiomrl.runtime.controls import build_control_callbacks
from axiomrl.runtime.run_utils import RunArtifacts, create_training_run, resolve_device
from axiomrl.runtime.trainer import TrainerState


@dataclass(slots=True)
class TrainingSession:
    device: torch.device
    run_artifacts: RunArtifacts
    callback_list: CallbackList
    trainer_state: TrainerState

    @property
    def run_context(self):
        return self.run_artifacts.run_context

    @property
    def logger(self):
        return self.run_artifacts.logger

    def close(self) -> None:
        self.run_artifacts.close()


def create_training_session(
    config: TrainConfig,
    *,
    algorithm: str,
    run_suffix: str | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainingSession:
    run_artifacts = create_training_run(config, run_suffix=run_suffix)
    return TrainingSession(
        device=resolve_device(config.device),
        run_artifacts=run_artifacts,
        callback_list=CallbackList(merge_callbacks(build_control_callbacks(config), callbacks)),
        trainer_state=TrainerState(algorithm=algorithm, run_dir=run_artifacts.run_context.run_dir),
    )
