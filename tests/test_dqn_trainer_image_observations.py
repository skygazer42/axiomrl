from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.workflows import evaluate_checkpoint


class DummyPixelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        obs_shape: tuple[int, int, int] = (4, 84, 84),
        action_dim: int = 4,
        episode_length: int = 8,
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
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, {}

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
        ("double_dqn", {}),
        ("dueling_dqn", {}),
        ("prioritized_dqn", {}),
        ("n_step_dqn", {"n_step": 3}),
        ("noisy_dqn", {"epsilon_start": 0.0, "epsilon_end": 0.0, "exploration_fraction": 0.0}),
        (
            "rainbow_dqn",
            {
                "n_step": 3,
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "exploration_fraction": 0.0,
                "prioritized_alpha": 0.6,
                "prioritized_beta_start": 0.4,
            },
        ),
        ("c51_dqn", {"v_min": -10.0, "v_max": 10.0, "num_atoms": 51}),
        ("qr_dqn", {"num_quantiles": 21, "kappa": 1.0}),
        ("iqn", {"num_quantiles": 8, "embedding_dim": 16, "kappa": 1.0}),
        (
            "fqf",
            {
                "num_quantiles": 8,
                "embedding_dim": 16,
                "kappa": 1.0,
                "entropy_coef": 1e-3,
                "fraction_learning_rate": 5e-4,
            },
        ),
    ],
)
def test_dqn_family_train_and_evaluate_with_image_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algo: str,
    algo_kwargs: dict[str, object],
) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo=algo,
        env_id="DummyPixelEnv-v0",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=0,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 8,
            "learning_starts": 8,
            "train_frequency": 1,
            "target_update_interval": 8,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 0.2,
            "epsilon_end": 0.1,
            "exploration_fraction": 0.5,
            "head_hidden_sizes": (64,),
            "features_dim": 64,
            **algo_kwargs,
        },
    )

    result = train_dqn(config, run_suffix=f"{algo}-pixels-smoke")
    assert result.checkpoint_path is not None
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert "eval_return_mean" in metrics

