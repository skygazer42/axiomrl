from pathlib import Path

import gymnasium as gym
import numpy as np

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.dqn_trainer import train_dqn
from axiomrl.runtime.workflows import evaluate_checkpoint


class TinyImageDiscreteEnv(gym.Env):
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
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step + int(action), dtype=np.uint8)
        terminated = self._step >= 4
        truncated = False
        reward = float(action == 1)
        return obs, reward, terminated, truncated, {}


def _register_tiny_image_env() -> str:
    env_id = "RLTrainingTest/AtariLikeImageDQN-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(
            id=env_id,
            entry_point="tests.support.envs:TinyImageDiscreteEnv",
            kwargs={"num_actions": 2, "reward_action": 1},
        )
    return env_id


def test_train_dqn_supports_image_observations_and_checkpoint_eval(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id=_register_tiny_image_env(),
        seed=23,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "head_hidden_sizes": (64,),
            "features_dim": 128,
        },
    )

    result = train_dqn(config, run_suffix="image-smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 96
    assert "eval_return_mean" in metrics
