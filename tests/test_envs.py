from pathlib import Path

import gymnasium as gym

from rl_training.envs.factory import make_vector_env
from rl_training.experiment.config import TrainConfig


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
