from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.evaluation_runner import EvaluationRunner
from axiomrl.runtime.types import MetricDict

DiscreteActionFn = Callable[[torch.Tensor], int]
ContinuousActionFn = Callable[[torch.Tensor], np.ndarray]


def evaluate_discrete_episodes(
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
    action_fn: DiscreteActionFn,
) -> MetricDict:
    return (
        EvaluationRunner(
            config=config,
            device=device,
            action_fn=action_fn,
        )
        .evaluate(num_episodes=num_episodes)
        .metrics
    )


def evaluate_continuous_episodes(
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
    action_fn: ContinuousActionFn,
) -> MetricDict:
    return (
        EvaluationRunner(
            config=config,
            device=device,
            action_fn=action_fn,
        )
        .evaluate(num_episodes=num_episodes)
        .metrics
    )
