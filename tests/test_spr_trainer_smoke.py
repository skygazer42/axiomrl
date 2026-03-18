from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.workflows import evaluate_checkpoint


class DummyPixelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        obs_shape: tuple[int, int, int] = (4, 84, 84),
        action_dim: int = 4,
        episode_length: int = 8,
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


def test_train_spr_supports_image_observations_and_representation_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="spr",
        env_id="DummyPixelEnv-v0",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 8,
            "learning_starts": 8,
            "train_frequency": 1,
            "target_update_interval": 8,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 0.2,
            "epsilon_end": 0.1,
            "exploration_fraction": 0.5,
            "head_hidden_sizes": (32,),
            "features_dim": 64,
            "spr_hidden_size": 64,
            "spr_projection_dim": 32,
            "spr_loss_coef": 0.5,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="spr-pixels-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 32
    assert "eval_return_mean" in result.metrics
    assert "spr_loss" in result.metrics
    assert "spr_cosine_similarity" in result.metrics

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics
