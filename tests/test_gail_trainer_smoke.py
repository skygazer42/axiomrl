from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.gail_trainer import train_gail


class TinyRenderDiscreteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
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
        self._state = np.array([action_int / 2.0, self._step / 4.0], dtype=np.float32)
        terminated = self._step >= 4
        truncated = False
        reward = 1.0 if action_int == (self._step % self.action_space.n) else 0.0
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        canvas[..., 0] = np.uint8(self._step * 32)
        canvas[16:80, 16:80, 1] = np.uint8(np.clip(self._state[0] * 255, 0, 255))
        canvas[32:64, 32:64, 2] = np.uint8(np.clip(self._state[1] * 255, 0, 255))
        return canvas


def _register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/GAILTrainerPixelsSmoke-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point=TinyRenderDiscreteEnv)
    return env_id


def test_train_gail_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="gail",
        env_id="CartPole-v1",
        seed=11,
        total_timesteps=256,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 64,
            "hidden_sizes": (64, 64),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "discriminator_learning_rate": 3e-4,
            "discriminator_updates": 4,
            "discriminator_batch_size": 64,
            "expert_dataset_kind": "random",
            "expert_dataset_size": 256,
            "expert_dataset_seed": 17,
        },
    )

    result = train_gail(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 256
    assert "eval_return_mean" in result.metrics


def test_train_gail_supports_pixel_observations(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="gail",
        env_id=_register_tiny_render_env(),
        seed=13,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 64,
            "head_hidden_sizes": (64,),
            "features_dim": 64,
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "discriminator_learning_rate": 3e-4,
            "discriminator_updates": 2,
            "discriminator_batch_size": 32,
            "expert_dataset_kind": "random",
            "expert_dataset_size": 128,
            "expert_dataset_seed": 19,
            "discriminator_head_hidden_sizes": (64,),
            "discriminator_features_dim": 64,
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

    result = train_gail(config, run_suffix="pixels-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics

