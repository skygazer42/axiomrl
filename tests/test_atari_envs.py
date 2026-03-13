from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_training.envs.atari import (
    ChannelFirstObservation,
    apply_atari_wrappers,
    resolve_atari_wrapper_config,
    split_env_kwargs,
)
from rl_training.envs.factory import build_env
from rl_training.experiment.config import TrainConfig


class DummyImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
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
        terminated = self._step >= 1
        truncated = False
        return obs, 2.5, terminated, truncated, {}


class DummyVectorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        del action
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step, dtype=np.float32)
        terminated = self._step >= 1
        truncated = False
        return obs, 2.5, terminated, truncated, {}


class DummyAtariPreprocessing(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        noop_max: int,
        frame_skip: int,
        screen_size: int,
        terminal_on_life_loss: bool,
        grayscale_obs: bool,
        scale_obs: bool,
    ) -> None:
        del noop_max, frame_skip, terminal_on_life_loss
        super().__init__(env)
        if grayscale_obs:
            shape = (screen_size, screen_size)
        else:
            shape = (screen_size, screen_size, 3)
        dtype = np.float32 if scale_obs else np.uint8
        high = 1.0 if scale_obs else 255
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=shape, dtype=dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)


def test_split_env_kwargs_separates_wrapper_config() -> None:
    env_kwargs, wrapper_kwargs = split_env_kwargs(
        {
            "render_mode": "rgb_array",
            "frameskip": 1,
            "wrappers": {"atari": {"frame_stack": 4}},
        }
    )

    assert env_kwargs == {"render_mode": "rgb_array", "frameskip": 1}
    assert wrapper_kwargs == {"atari": {"frame_stack": 4}}


def test_resolve_atari_wrapper_config_uses_tags_and_eval_mode() -> None:
    train_config = resolve_atari_wrapper_config(
        env_id="CartPole-v1",
        tags=("atari",),
        wrapper_kwargs={},
        evaluation=False,
    )
    eval_config = resolve_atari_wrapper_config(
        env_id="CartPole-v1",
        tags=("atari",),
        wrapper_kwargs={},
        evaluation=True,
    )

    assert train_config is not None
    assert eval_config is not None
    assert train_config.clip_reward is True
    assert eval_config.clip_reward is False


def test_channel_first_observation_converts_hwc_to_chw() -> None:
    env = ChannelFirstObservation(DummyImageEnv())
    obs, _ = env.reset()

    assert obs.shape == (3, 210, 160)

    env.close()


def test_build_env_applies_atari_wrappers_when_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyImageEnv())
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="ppo",
        env_id="ALE/Breakout-v5",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
        env_kwargs={
            "wrappers": {
                "atari": {
                    "screen_size": 84,
                    "frame_skip": 4,
                    "grayscale_obs": True,
                    "frame_stack": 4,
                    "clip_reward": True,
                    "channel_first": True,
                }
            }
        },
    )

    env = build_env(config, env_index=0)
    obs, _ = env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert obs.shape == (4, 84, 84)
    assert reward == pytest.approx(1.0)
    assert terminated or truncated

    env.close()


def test_apply_atari_wrappers_leaves_non_atari_env_unchanged() -> None:
    env = DummyVectorEnv()
    wrapped = apply_atari_wrappers(env, atari_config=None)
    obs, _ = wrapped.reset()

    assert wrapped is env
    assert obs.shape == (4,)

    wrapped.close()
