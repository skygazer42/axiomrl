from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.registry import get_algorithm_spec
from axiomrl.runtime.workflows import evaluate_checkpoint


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


@pytest.mark.parametrize(
    ("algo", "algo_kwargs"),
    [
        (
            "a2c",
            {
                "num_steps": 4,
                "learning_rate": 1e-3,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "features_dim": 32,
                "head_hidden_sizes": (64,),
            },
        ),
        (
            "impala",
            {
                "num_steps": 4,
                "learning_rate": 1e-3,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "gamma": 0.99,
                "rho_clip": 1.0,
                "c_clip": 1.0,
                "pg_rho_clip": 1.0,
                "max_grad_norm": 0.5,
                "features_dim": 32,
                "head_hidden_sizes": (64,),
            },
        ),
        (
            "ppg",
            {
                "num_steps": 4,
                "update_epochs": 1,
                "minibatch_size": 8,
                "learning_rate": 1e-3,
                "aux_learning_rate": 1e-3,
                "clip_coef": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
                "aux_frequency": 1,
                "aux_epochs": 1,
                "aux_minibatch_size": 8,
                "aux_buffer_rollouts": 1,
                "aux_value_coef": 1.0,
                "behavior_clone_coef": 1.0,
                "value_clone_coef": 1.0,
                "features_dim": 32,
                "head_hidden_sizes": (64,),
            },
        ),
    ],
)
def test_train_onpolicy_algorithms_support_image_observations(
    algo: str,
    algo_kwargs: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo=algo,
        env_id="DummyPixelEnv-v0",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        device="cpu",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs=algo_kwargs,
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix=f"{algo}-pixel-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 32
    assert "eval_return_mean" in result.metrics

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics
