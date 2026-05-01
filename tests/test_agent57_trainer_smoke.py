from pathlib import Path

import gymnasium as gym
import numpy as np

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


def test_train_agent57_supports_image_observations_and_intrinsic_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gym, "make", lambda env_id, **kwargs: DummyPixelEnv())

    config = TrainConfig(
        algo="agent57",
        env_id="DummyPixelEnv-v0",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        device="cpu",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 64,
            "batch_size": 2,
            "learning_starts": 4,
            "train_frequency": 1,
            "target_update_interval": 8,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
            "prioritized_beta_end": 1.0,
            "prioritized_beta_fraction": 1.0,
            "priority_eta": 0.9,
            "n_step": 2,
            "intrinsic_reward_coef": 0.5,
            "rnd_learning_rate": 1e-3,
            "rnd_hidden_sizes": (32,),
            "rnd_embedding_dim": 32,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="agent57-pixel-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 32
    assert "eval_return_mean" in result.metrics
    assert "intrinsic_reward_mean" in result.metrics
    assert "rnd_loss" in result.metrics

    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)
    assert "eval_return_mean" in metrics
