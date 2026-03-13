from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.envs.factory import make_vector_env
from rl_training.experiment.config import TrainConfig


class TinyImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        del action
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step, dtype=np.uint8)
        terminated = self._step >= 4
        truncated = False
        return obs, 1.0, terminated, truncated, {}


class TinyRenderContinuousEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0
        self._state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        self._state.fill(0.0)
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action_value = float(np.asarray(action).reshape(-1)[0])
        self._step += 1
        self._state = np.array([action_value, self._step / 4.0, -action_value], dtype=np.float32)
        terminated = self._step >= 4
        truncated = False
        reward = 1.0 - abs(action_value)
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        action_intensity = int(np.clip((self._state[0] + 1.0) * 127.5, 0, 255))
        canvas[..., 0] = np.uint8(self._step * 32)
        canvas[16:80, 16:80, 1] = np.uint8(action_intensity)
        canvas[32:64, 32:64, 2] = np.uint8(255 - action_intensity)
        return canvas


def _register_tiny_image_env() -> str:
    env_id = "RLTrainingTest/ImageEnv-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point=TinyImageEnv)
    return env_id


def _register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/RenderContinuousEnv-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point=TinyRenderContinuousEnv)
    return env_id


def test_make_vector_env_returns_sync_vector_env(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=11,
        total_timesteps=256,
        output_dir=tmp_path,
        num_envs=4,
    )

    envs = make_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)

    assert isinstance(envs, gym.vector.SyncVectorEnv)
    assert obs.shape[0] == 4

    envs.close()


def test_make_vector_env_preserves_image_observation_shape(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id=_register_tiny_image_env(),
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
    )

    envs = make_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)

    assert isinstance(envs, gym.vector.SyncVectorEnv)
    assert obs.shape == (2, 4, 84, 84)

    envs.close()


def test_make_vector_env_supports_reward_wrapper_config(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        env_kwargs={
            "wrappers": {
                "reward": {
                    "scale": 0.5,
                    "clip": [-1.0, 1.0],
                }
            }
        },
    )

    envs = make_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)

    assert obs.shape[0] == 2

    envs.close()


def test_make_vector_env_supports_pixel_wrapper_config(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=19,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
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

    envs = make_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)

    assert isinstance(envs, gym.vector.SyncVectorEnv)
    assert obs.shape == (2, 9, 84, 84)

    envs.close()
