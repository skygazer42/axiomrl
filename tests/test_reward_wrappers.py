from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.envs import apply_reward_wrappers, resolve_reward_preset, resolve_reward_wrapper_config
from rl_training.envs.factory import build_env
from rl_training.experiment.config import TrainConfig


class DummyRewardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros((3,), dtype=np.float32), {}

    def step(self, action: int):
        del action
        self._step += 1
        terminated = self._step >= 1
        return np.zeros((3,), dtype=np.float32), 2.0, terminated, False, {}


def test_resolve_reward_wrapper_config_accepts_scale_shift_and_clip() -> None:
    config = resolve_reward_wrapper_config(
        {
            "reward": {
                "scale": 0.5,
                "shift": 1.0,
                "clip": [-1.0, 1.5],
            }
        }
    )

    assert config is not None
    assert config.scale == 0.5
    assert config.shift == 1.0
    assert config.clip_min == -1.0
    assert config.clip_max == 1.5


def test_resolve_reward_wrapper_config_supports_named_preset() -> None:
    config = resolve_reward_wrapper_config({"reward": {"preset": "sparse_goal_zero_one"}})

    assert config is not None
    assert config.shift == 1.0
    assert config.clip_min == 0.0
    assert config.clip_max == 1.0


def test_resolve_reward_preset_returns_sign_clip_transform() -> None:
    config = resolve_reward_preset("sign_clip")

    assert config.sign is True
    assert config.scale == 1.0


def test_apply_reward_wrappers_scales_shifts_and_clips_reward() -> None:
    env = apply_reward_wrappers(
        DummyRewardEnv(),
        resolve_reward_wrapper_config({"reward": {"scale": 0.5, "shift": 1.0, "clip": [0.0, 1.5]}}),
    )
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == 1.5
    assert terminated is True
    assert truncated is False

    env.close()


def test_apply_reward_wrappers_supports_sign_clip_preset() -> None:
    env = apply_reward_wrappers(
        DummyRewardEnv(),
        resolve_reward_wrapper_config({"reward": {"preset": "sign_clip"}}),
    )
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == 1.0
    assert terminated is True
    assert truncated is False

    env.close()


def test_build_env_applies_generic_reward_wrapper_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyRewardEnv())

    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=5,
        total_timesteps=16,
        output_dir=tmp_path,
        env_kwargs={
            "wrappers": {
                "reward": {
                    "scale": 2.0,
                    "shift": -1.0,
                    "clip_min": 0.0,
                    "clip_max": 2.5,
                }
            }
        },
    )

    env = build_env(config, env_index=0)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert reward == 2.5
    assert terminated or truncated

    env.close()


def test_build_env_applies_reward_preset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyRewardEnv())

    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=16,
        output_dir=tmp_path,
        env_kwargs={
            "wrappers": {
                "reward": {
                    "preset": "clip_1",
                }
            }
        },
    )

    env = build_env(config, env_index=0)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert reward == 1.0
    assert terminated or truncated

    env.close()
