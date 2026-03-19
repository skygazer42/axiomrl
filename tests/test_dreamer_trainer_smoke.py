from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dreamer_trainer import train_dreamer


class TinyRenderDiscreteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0
        self._state = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        self._state.fill(0.0)
        return self._state.copy(), {}

    def step(self, action: int):
        action_int = int(action)
        self._step += 1
        self._state = np.array([action_int / 3.0, self._step / 6.0], dtype=np.float32)
        terminated = self._step >= 6
        truncated = False
        reward = 1.0 if action_int == (self._step % self.action_space.n) else 0.0
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        canvas[..., 0] = np.uint8(self._step * 24)
        canvas[16:80, 16:80, 1] = np.uint8(np.clip(self._state[0] * 255, 0, 255))
        canvas[32:64, 32:64, 2] = np.uint8(np.clip(self._state[1] * 255, 0, 255))
        return canvas


def _register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/DreamerTrainerSmoke-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderDiscreteEnv")
    return env_id


def test_train_dreamer_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dreamer",
        env_id=_register_tiny_render_env(),
        seed=101,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 32,
            "train_frequency": 1,
            "world_model_updates": 1,
            "imagination_batch_size": 8,
            "imagination_horizon": 3,
            "features_dim": 64,
            "action_embed_dim": 16,
            "world_model_learning_rate": 1e-3,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 1e-3,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    result = train_dreamer(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_dreamer_supports_entropy_coefficient_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dreamer",
        env_id=_register_tiny_render_env(),
        seed=111,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 32,
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
            "entropy_coef_schedule": {
                "type": "linear",
                "start": 1e-3,
                "end": 1e-4,
            },
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    result = train_dreamer(config, run_suffix="entropy-schedule-smoke")

    assert result.checkpoint_path is not None
    assert result.metrics["entropy_coef"] == pytest.approx(1e-4, rel=1e-6)
