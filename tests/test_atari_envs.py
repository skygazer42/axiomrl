import sys
from pathlib import Path
from types import ModuleType

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
from rl_training.envs.tennis_events import (
    TennisEventConfig,
    TennisEventRewardWrapper,
    resolve_tennis_event_wrapper_config,
)
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


class DummyStackedTennisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frames: list[np.ndarray], rewards: list[float], terminated_at: int | None = None) -> None:
        super().__init__()
        self.frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        self.rewards = list(rewards)
        self.terminated_at = terminated_at
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.frames[0].shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._index = 0
        return self.frames[0], {}

    def step(self, action: int):
        del action
        self._index += 1
        frame = self.frames[min(self._index, len(self.frames) - 1)]
        reward = float(self.rewards[min(self._index - 1, len(self.rewards) - 1)])
        terminated = self.terminated_at is not None and self._index >= self.terminated_at
        truncated = False
        return frame, reward, terminated, truncated, {}


def _stacked_ball_frame(*xs: int, y: int = 40) -> np.ndarray:
    frame = np.zeros((4, 84, 84), dtype=np.uint8)
    for channel, x in enumerate(xs[-4:]):
        frame[channel, y, x] = 255
    return frame


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


def test_resolve_tennis_event_wrapper_config_reads_mapping() -> None:
    config = resolve_tennis_event_wrapper_config(
        {
            "tennis_events": {
                "rally_survival_bonus": 0.001,
                "net_cross_bonus": 0.05,
                "successful_return_bonus": 0.1,
                "failure_penalty": -0.2,
                "deep_landing_bonus": 0.03,
                "wide_landing_bonus": 0.02,
            }
        }
    )

    assert config == TennisEventConfig(
        rally_survival_bonus=pytest.approx(0.001),
        net_cross_bonus=pytest.approx(0.05),
        successful_return_bonus=pytest.approx(0.1),
        failure_penalty=pytest.approx(-0.2),
        deep_landing_bonus=pytest.approx(0.03),
        wide_landing_bonus=pytest.approx(0.02),
    )


def test_tennis_event_reward_wrapper_awards_net_cross_and_failure_penalty() -> None:
    env = DummyStackedTennisEnv(
        frames=[
            _stacked_ball_frame(10, 10, 10, 10),
            _stacked_ball_frame(20, 20, 20, 20),
            _stacked_ball_frame(60, 60, 60, 60),
            _stacked_ball_frame(62, 62, 62, 62),
        ],
        rewards=[0.0, 0.0, -1.0],
        terminated_at=3,
    )
    wrapped = TennisEventRewardWrapper(
        env,
        TennisEventConfig(
            rally_survival_bonus=0.001,
            net_cross_bonus=0.05,
            successful_return_bonus=0.1,
            failure_penalty=-0.2,
            agent_side="left",
        ),
    )

    wrapped.reset()
    _, reward_one, _, _, _ = wrapped.step(0)
    _, reward_two, _, _, _ = wrapped.step(0)
    _, reward_three, terminated, _, _ = wrapped.step(0)

    assert reward_one == pytest.approx(0.001)
    assert reward_two == pytest.approx(0.151)
    assert terminated is True
    assert reward_three == pytest.approx(-1.199)


def test_tennis_event_reward_wrapper_awards_offensive_landing_bonuses() -> None:
    env = DummyStackedTennisEnv(
        frames=[
            _stacked_ball_frame(12, 12, 12, 12, y=42),
            _stacked_ball_frame(20, 20, 20, 20, y=42),
            _stacked_ball_frame(74, 74, 74, 74, y=8),
        ],
        rewards=[0.0, 0.0],
        terminated_at=None,
    )
    wrapped = TennisEventRewardWrapper(
        env,
        TennisEventConfig(
            rally_survival_bonus=0.001,
            net_cross_bonus=0.05,
            successful_return_bonus=0.1,
            deep_landing_bonus=0.03,
            wide_landing_bonus=0.02,
            failure_penalty=-0.2,
            agent_side="left",
        ),
    )

    wrapped.reset()
    _, reward_one, _, _, _ = wrapped.step(0)
    _, reward_two, _, _, _ = wrapped.step(0)

    assert reward_one == pytest.approx(0.001)
    assert reward_two == pytest.approx(0.201)


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


def test_resolve_atari_wrapper_config_defers_to_generic_reward_strategy_by_default() -> None:
    train_config = resolve_atari_wrapper_config(
        env_id="ALE/Breakout-v5",
        tags=("atari",),
        wrapper_kwargs={},
        evaluation=False,
        reward_wrapper_active=True,
    )

    assert train_config is not None
    assert train_config.clip_reward is False


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


def test_build_env_registers_ale_envs_before_retrying_make(monkeypatch, tmp_path: Path) -> None:
    fake_ale = ModuleType("ale_py")
    register_calls: list[ModuleType] = []
    state = {"registered": False}

    def fake_register_envs(module: ModuleType) -> None:
        register_calls.append(module)
        state["registered"] = True

    def fake_make(env_id: str, **kwargs):
        del kwargs
        if env_id == "ALE/Tennis-v5" and not state["registered"]:
            raise gym.error.NameNotFound("ALE/Tennis-v5")
        return DummyImageEnv()

    monkeypatch.setitem(sys.modules, "ale_py", fake_ale)
    monkeypatch.setattr(gym, "register_envs", fake_register_envs, raising=False)
    monkeypatch.setattr(gym, "make", fake_make)
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="ppo",
        env_id="ALE/Tennis-v5",
        seed=5,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
    )

    env = build_env(config, env_index=0)

    assert register_calls == [fake_ale]

    env.close()


def test_build_env_prefers_generic_reward_strategy_over_default_atari_clip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyImageEnv())
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="dqn",
        env_id="ALE/Breakout-v5",
        seed=11,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
        env_kwargs={
            "wrappers": {
                "reward": {
                    "scale": 0.5,
                }
            }
        },
    )

    env = build_env(config, env_index=0)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert reward == pytest.approx(1.25)
    assert terminated or truncated

    env.close()


def test_build_env_keeps_explicit_atari_clip_reward_with_generic_reward_strategy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyImageEnv())
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="dqn",
        env_id="ALE/Breakout-v5",
        seed=13,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
        env_kwargs={
            "wrappers": {
                "atari": {
                    "clip_reward": True,
                },
                "reward": {
                    "scale": 0.5,
                },
            }
        },
    )

    env = build_env(config, env_index=0)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert reward == pytest.approx(0.5)
    assert terminated or truncated

    env.close()


def test_build_env_applies_evaluation_env_overrides_for_atari_protocol(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    def _make_env(env_id: str, **kwargs):
        del env_id
        captured_kwargs.update(kwargs)
        return DummyImageEnv()

    monkeypatch.setattr(gym, "make", _make_env)
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="dqn",
        env_id="ALE/Breakout-v5",
        seed=17,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "wrappers": {
                "atari": {
                    "screen_size": 84,
                    "frame_skip": 4,
                    "noop_max": 30,
                    "grayscale_obs": True,
                    "frame_stack": 4,
                    "clip_reward": True,
                    "channel_first": True,
                }
            },
            "evaluation": {
                "repeat_action_probability": 0.25,
                "wrappers": {
                    "atari": {
                        "clip_reward": False,
                    }
                },
            },
        },
    )

    env = build_env(config, env_index=0, evaluation=True)
    obs, _ = env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert captured_kwargs["frameskip"] == 1
    assert captured_kwargs["repeat_action_probability"] == pytest.approx(0.25)
    assert obs.shape == (4, 84, 84)
    assert reward == pytest.approx(2.5)
    assert terminated or truncated

    env.close()


def test_build_env_applies_training_env_overrides_for_atari_protocol(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    def _make_env(env_id: str, **kwargs):
        del env_id
        captured_kwargs.update(kwargs)
        return DummyImageEnv()

    monkeypatch.setattr(gym, "make", _make_env)
    monkeypatch.setattr(gym.wrappers, "AtariPreprocessing", DummyAtariPreprocessing)

    config = TrainConfig(
        algo="dqn",
        env_id="ALE/Breakout-v5",
        seed=19,
        total_timesteps=32,
        output_dir=tmp_path,
        tags=("atari",),
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": 0.25,
            "wrappers": {
                "atari": {
                    "screen_size": 84,
                    "frame_skip": 4,
                    "noop_max": 30,
                    "grayscale_obs": True,
                    "frame_stack": 4,
                    "clip_reward": True,
                    "channel_first": True,
                }
            },
            "training": {
                "repeat_action_probability": 0.0,
            },
        },
    )

    env = build_env(config, env_index=0, evaluation=False)
    env.reset(seed=config.seed)
    _, reward, terminated, truncated, _ = env.step(1)

    assert captured_kwargs["repeat_action_probability"] == pytest.approx(0.0)
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
