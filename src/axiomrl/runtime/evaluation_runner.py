from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from axiomrl.envs.factory import build_env
from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.evaluator import EvalResult

ActionFn = Callable[[torch.Tensor], object]


@dataclass(slots=True)
class EvaluationRunner:
    config: TrainConfig
    device: torch.device
    action_fn: ActionFn

    def evaluate(self, *, num_episodes: int) -> EvalResult:
        env = build_env(self.config, 0, evaluation=True)
        returns: list[float] = []
        episode_metric_totals: dict[str, float] = {}
        episode_metric_counts: dict[str, int] = {}

        try:
            bind_env = getattr(self.action_fn, "bind_env", None)
            if callable(bind_env):
                bind_env(env)
            for episode_index in range(num_episodes):
                reset_action_fn = getattr(self.action_fn, "reset", None)
                if callable(reset_action_fn):
                    reset_action_fn()
                obs, _ = env.reset(seed=self.config.seed + episode_index)
                done = False
                truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    prepare_observation = getattr(self.action_fn, "prepare_observation", None)
                    if callable(prepare_observation):
                        action_input = prepare_observation(obs)
                    else:
                        action_input = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    action = self.action_fn(action_input)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_return += float(reward)
                    after_step = getattr(self.action_fn, "after_step", None)
                    if callable(after_step):
                        after_step(obs, float(reward), bool(done), bool(truncated), info)

                returns.append(episode_return)
                episode_metrics = getattr(self.action_fn, "episode_metrics", None)
                if callable(episode_metrics):
                    for key, value in episode_metrics().items():
                        episode_metric_totals[key] = episode_metric_totals.get(key, 0.0) + float(value)
                        episode_metric_counts[key] = episode_metric_counts.get(key, 0) + 1
        finally:
            env.close()

        metrics = {
            "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
            "eval_return_std": float(np.std(returns)) if returns else 0.0,
            "eval_episodes": float(len(returns)),
        }
        for key, total in episode_metric_totals.items():
            count = episode_metric_counts.get(key, 0)
            metrics[key] = float(total / count) if count > 0 else 0.0

        return EvalResult(
            num_episodes=len(returns),
            metrics=metrics,
        )
