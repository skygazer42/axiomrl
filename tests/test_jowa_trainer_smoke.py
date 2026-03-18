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


def test_train_jowa_writes_checkpoint_and_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="jowa",
        env_id="DummyPixelEnv-v0",
        seed=151,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 8,
            "learning_starts": 8,
            "train_frequency": 1,
            "target_update_interval": 8,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 0.2,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "features_dim": 64,
            "head_hidden_sizes": [64],
            "jowa_transition_hidden_size": 64,
            "jowa_action_embed_dim": 16,
            "jowa_reward_hidden_size": 64,
            "jowa_world_model_loss_coef": 1.0,
            "jowa_reward_loss_coef": 1.0,
            "jowa_reconstruction_loss_coef": 1.0,
            "jowa_consistency_loss_coef": 0.5,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="jowa-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics
    assert "jowa_model_loss" in result.metrics
    assert "jowa_reconstruction_loss" in result.metrics
    assert "jowa_consistency_loss" in result.metrics

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics
