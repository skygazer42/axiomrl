from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


TrainFn = Callable[..., TrainResult]
EvaluateFn = Callable[[TrainConfig, CheckpointState, torch.device, int], MetricDict]
PredictFn = Callable[[TrainConfig, CheckpointState, torch.device, object, bool], int | np.ndarray]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    train_fn: TrainFn
    evaluate_fn: EvaluateFn
    predict_fn: PredictFn
