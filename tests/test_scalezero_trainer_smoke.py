from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

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


def test_train_scalezero_writes_checkpoint_and_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="scalezero",
        env_id="DummyPixelEnv-v0",
        seed=171,
        total_timesteps=20,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 64,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "temperature": 1.0,
            "num_simulations": 5,
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
            "num_experts": 3,
            "gating_hidden_size": 32,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="scalezero-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 20
    assert "eval_return_mean" in result.metrics
    assert "scalezero_loss" in result.metrics
    assert "scalezero_moe_entropy" in result.metrics
    assert result.metrics["scalezero_num_experts"] == 3.0

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics


def test_train_scalezero_supports_temperature_schedule(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="scalezero",
        env_id="DummyPixelEnv-v0",
        seed=173,
        total_timesteps=20,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 64,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "temperature_schedule": {
                "type": "linear",
                "start": 1.3,
                "end": 0.25,
            },
            "num_simulations": 5,
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
            "num_experts": 3,
            "gating_hidden_size": 32,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="scalezero-temp-schedule")

    assert result.checkpoint_path is not None
    assert result.metrics["temperature"] == pytest.approx(0.25, rel=1e-6)


def test_train_scalezero_supports_root_exploration_fraction_schedule(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="scalezero",
        env_id="DummyPixelEnv-v0",
        seed=175,
        total_timesteps=20,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 64,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "temperature": 1.0,
            "root_exploration_fraction_schedule": {
                "type": "linear",
                "start": 0.5,
                "end": 0.1,
            },
            "num_simulations": 5,
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
            "num_experts": 3,
            "gating_hidden_size": 32,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="scalezero-root-noise-schedule")

    assert result.checkpoint_path is not None
    assert result.metrics["root_exploration_fraction"] == pytest.approx(0.1, rel=1e-6)


def test_train_scalezero_supports_num_simulations_schedule(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="scalezero",
        env_id="DummyPixelEnv-v0",
        seed=177,
        total_timesteps=20,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 64,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "unroll_steps": 2,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "temperature": 1.0,
            "num_simulations_schedule": {
                "type": "linear",
                "start": 9,
                "end": 5,
            },
            "latent_dim": 32,
            "action_embed_dim": 16,
            "dynamics_hidden_sizes": (32,),
            "prediction_hidden_sizes": (32,),
            "num_experts": 3,
            "gating_hidden_size": 32,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="scalezero-sims-schedule")

    assert result.checkpoint_path is not None
    assert result.metrics["num_simulations"] == 5.0
