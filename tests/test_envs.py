from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.envs.factory import build_env, make_vector_env, resolve_mode_env_kwargs
from rl_training.envs.video import resolve_video_wrapper_config
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


class DummyRecordVideo(gym.Wrapper):
    last_init: dict[str, object] | None = None

    def __init__(
        self,
        env: gym.Env,
        *,
        video_folder: str,
        episode_trigger=None,
        step_trigger=None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
    ) -> None:
        super().__init__(env)
        DummyRecordVideo.last_init = {
            "video_folder": video_folder,
            "episode_trigger": episode_trigger,
            "step_trigger": step_trigger,
            "video_length": video_length,
            "name_prefix": name_prefix,
            "fps": fps,
            "disable_logger": disable_logger,
        }


def _register_tiny_image_env() -> str:
    env_id = "RLTrainingTest/ImageEnv-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyImageEnv")
    return env_id


def _register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/RenderContinuousEnv-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderContinuousEnv")
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


def test_make_vector_env_returns_async_vector_env_when_requested(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=256,
        output_dir=tmp_path,
        execution_backend="local_async",
        num_envs=2,
    )

    envs = make_vector_env(config)
    obs, _ = envs.reset(seed=config.seed)

    assert isinstance(envs, gym.vector.AsyncVectorEnv)
    assert obs.shape[0] == 2

    envs.close()


def test_make_vector_env_supports_parent_registered_custom_env_with_async_backend(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=21,
        total_timesteps=64,
        output_dir=tmp_path,
        execution_backend="local_async",
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

    assert isinstance(envs, gym.vector.AsyncVectorEnv)
    assert obs.shape == (2, 9, 84, 84)

    envs.close()


def test_resolve_mode_env_kwargs_merges_evaluation_overrides_recursively() -> None:
    resolved = resolve_mode_env_kwargs(
        {
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "wrappers": {
                "atari": {
                    "frame_stack": 4,
                    "clip_reward": True,
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
        evaluation=True,
    )

    assert resolved["frameskip"] == 1
    assert resolved["repeat_action_probability"] == 0.25
    assert resolved["wrappers"]["atari"]["frame_stack"] == 4
    assert resolved["wrappers"]["atari"]["clip_reward"] is False
    assert "evaluation" not in resolved


def test_resolve_mode_env_kwargs_merges_training_overrides_recursively() -> None:
    resolved = resolve_mode_env_kwargs(
        {
            "frameskip": 1,
            "repeat_action_probability": 0.25,
            "wrappers": {
                "atari": {
                    "frame_stack": 4,
                }
            },
            "training": {
                "repeat_action_probability": 0.0,
                "wrappers": {
                    "atari": {
                        "terminal_on_life_loss": True,
                    }
                },
            },
        },
        evaluation=False,
    )

    assert resolved["frameskip"] == 1
    assert resolved["repeat_action_probability"] == 0.0
    assert resolved["wrappers"]["atari"]["frame_stack"] == 4
    assert resolved["wrappers"]["atari"]["terminal_on_life_loss"] is True
    assert "training" not in resolved


def test_resolve_video_wrapper_config_supports_episode_trigger_settings() -> None:
    config = resolve_video_wrapper_config(
        {
            "video": {
                "episode_trigger_every": 2,
                "video_length": 128,
                "name_prefix": "eval-rollout",
            }
        }
    )

    assert config is not None
    assert config.episode_trigger_every == 2
    assert config.video_length == 128
    assert config.name_prefix == "eval-rollout"


def test_build_env_applies_evaluation_video_wrapper(monkeypatch, tmp_path: Path) -> None:
    DummyRecordVideo.last_init = None
    monkeypatch.setattr(gym.wrappers, "RecordVideo", DummyRecordVideo)

    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=23,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=1,
        env_kwargs={
            "evaluation": {
                "render_mode": "rgb_array",
                "wrappers": {
                    "video": {
                        "episode_trigger_every": 2,
                        "video_length": 32,
                        "name_prefix": "eval-rollout",
                    }
                },
            }
        },
    )

    env = build_env(config, env_index=0, evaluation=True)
    obs, _ = env.reset(seed=config.seed)

    assert obs.shape == (3,)
    assert DummyRecordVideo.last_init is not None
    assert Path(str(DummyRecordVideo.last_init["video_folder"])) == tmp_path / "videos" / "evaluation"
    assert DummyRecordVideo.last_init["video_length"] == 32
    assert DummyRecordVideo.last_init["name_prefix"] == "eval-rollout"
    assert DummyRecordVideo.last_init["episode_trigger"] is not None
    assert DummyRecordVideo.last_init["step_trigger"] is None
    assert DummyRecordVideo.last_init["episode_trigger"](0) is True
    assert DummyRecordVideo.last_init["episode_trigger"](1) is False
    assert DummyRecordVideo.last_init["episode_trigger"](2) is True

    env.close()


def test_build_env_does_not_apply_video_wrapper_during_training_when_only_evaluation_config_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    DummyRecordVideo.last_init = None
    monkeypatch.setattr(gym.wrappers, "RecordVideo", DummyRecordVideo)

    config = TrainConfig(
        algo="drqv2",
        env_id=_register_tiny_render_env(),
        seed=29,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=1,
        env_kwargs={
            "evaluation": {
                "render_mode": "rgb_array",
                "wrappers": {
                    "video": {
                        "episode_trigger_every": 1,
                    }
                },
            }
        },
    )

    env = build_env(config, env_index=0, evaluation=False)
    obs, _ = env.reset(seed=config.seed)

    assert obs.shape == (3,)
    assert DummyRecordVideo.last_init is None

    env.close()


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
