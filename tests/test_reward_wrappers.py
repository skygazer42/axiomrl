from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from axiomrl.envs import apply_reward_wrappers, resolve_reward_preset, resolve_reward_wrapper_config
from axiomrl.envs.factory import build_env
from axiomrl.experiment.config import TrainConfig


class DummyRewardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        reward: float = 2.0,
        info: dict[str, object] | None = None,
        terminated_after: int = 1,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.reward_value = float(reward)
        self.info = {} if info is None else dict(info)
        self.terminated_after = int(terminated_after)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros((3,), dtype=np.float32), {}

    def step(self, action: int):
        del action
        self._step += 1
        terminated = self._step >= self.terminated_after
        return np.zeros((3,), dtype=np.float32), self.reward_value, terminated, False, dict(self.info)


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
    assert config.scale == pytest.approx(0.5)
    assert config.shift == pytest.approx(1.0)
    assert config.clip_min == pytest.approx(-1.0)
    assert config.clip_max == pytest.approx(1.5)


def test_resolve_reward_wrapper_config_supports_named_preset() -> None:
    config = resolve_reward_wrapper_config({"reward": {"preset": "sparse_goal_zero_one"}})

    assert config is not None
    assert config.shift == pytest.approx(1.0)
    assert config.clip_min == pytest.approx(0.0)
    assert config.clip_max == pytest.approx(1.0)


def test_resolve_reward_wrapper_config_supports_strategy_alias_and_outcome_shaping() -> None:
    config = resolve_reward_wrapper_config(
        {
            "reward": {
                "strategy": "clip_1",
                "step_penalty": -0.05,
                "success_bonus": 1.0,
                "failure_penalty": -1.5,
            }
        }
    )

    assert config is not None
    assert config.clip_min == pytest.approx(-1.0)
    assert config.clip_max == pytest.approx(1.0)
    assert config.step_penalty == pytest.approx(-0.05)
    assert config.success_bonus == pytest.approx(1.0)
    assert config.failure_penalty == pytest.approx(-1.5)


def test_resolve_reward_preset_returns_sign_clip_transform() -> None:
    config = resolve_reward_preset("sign_clip")

    assert config.sign is True
    assert config.scale == pytest.approx(1.0)


def test_resolve_reward_preset_returns_atari_clip_transform() -> None:
    config = resolve_reward_preset("atari_clip")

    assert config.sign is True
    assert config.clip_min is None
    assert config.clip_max is None


def test_resolve_reward_preset_returns_survival_penalty_transform() -> None:
    config = resolve_reward_preset("survival_penalty")

    assert config.step_penalty == pytest.approx(-0.01)
    assert config.success_bonus == pytest.approx(0.0)


def test_resolve_reward_preset_returns_goal_success_bonus_transform() -> None:
    config = resolve_reward_preset("goal_success_bonus")

    assert config.success_bonus == pytest.approx(1.0)
    assert config.failure_penalty == pytest.approx(0.0)


def test_apply_reward_wrappers_scales_shifts_and_clips_reward() -> None:
    env = apply_reward_wrappers(
        DummyRewardEnv(),
        resolve_reward_wrapper_config({"reward": {"scale": 0.5, "shift": 1.0, "clip": [0.0, 1.5]}}),
    )
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == pytest.approx(1.5)
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

    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False

    env.close()


def test_apply_reward_wrappers_adds_step_penalty_and_success_bonus() -> None:
    env = apply_reward_wrappers(
        DummyRewardEnv(reward=0.25, info={"is_success": True}),
        resolve_reward_wrapper_config({"reward": {"step_penalty": -0.05, "success_bonus": 1.0}}),
    )
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == pytest.approx(1.2)
    assert terminated is True
    assert truncated is False

    env.close()


def test_apply_reward_wrappers_adds_failure_penalty_for_unsuccessful_episode() -> None:
    env = apply_reward_wrappers(
        DummyRewardEnv(reward=0.25, info={"is_success": False}),
        resolve_reward_wrapper_config({"reward": {"failure_penalty": -1.5}}),
    )
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)

    assert reward == pytest.approx(-1.25)
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

    assert reward == pytest.approx(2.5)
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

    assert reward == pytest.approx(1.0)
    assert terminated or truncated

    env.close()


def test_build_env_applies_reward_outcome_strategy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        gym,
        "make",
        lambda env_id, **kwargs: DummyRewardEnv(reward=0.5, info={"is_success": True}),
    )

    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=9,
        total_timesteps=16,
        output_dir=tmp_path,
        env_kwargs={
            "wrappers": {
                "reward": {
                    "step_penalty": -0.1,
                    "success_bonus": 2.0,
                }
            }
        },
    )

    env = build_env(config, env_index=0)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert reward == pytest.approx(2.4)
    assert terminated or truncated

    env.close()
