from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry import get_algorithm_spec
from axiomrl.runtime.workflows import evaluate_checkpoint


class DummyPixelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        obs_shape: tuple[int, int, int] = (4, 84, 84),
        action_dim: int = 4,
        episode_length: int = 6,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(int(action_dim))
        self._episode_length = int(episode_length)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step % 255, dtype=np.uint8)
        reward = float(int(action) == 0)
        terminated = self._step >= self._episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_train_eadream_writes_checkpoint_and_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="eadream",
        env_id="DummyPixelEnv-v0",
        seed=161,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 8,
            "learning_starts": 8,
            "train_frequency": 1,
            "world_model_updates": 1,
            "actor_critic_updates": 1,
            "imagination_batch_size": 8,
            "imagination_horizon": 3,
            "features_dim": 64,
            "action_embed_dim": 16,
            "world_model_learning_rate": 1e-3,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 1e-3,
            "event_hidden_sizes": (32,),
            "event_loss_coef": 0.5,
            "event_threshold": 0.01,
            "event_scale": 0.75,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="eadream-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics
    assert "eadream_world_model_loss" in result.metrics
    assert "eadream_event_loss" in result.metrics
    assert "eadream_event_rate" in result.metrics

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics
