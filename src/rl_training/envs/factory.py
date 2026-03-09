from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from rl_training.experiment.config import TrainConfig


EnvFactory = Callable[[], gym.Env]


def make_env(config: TrainConfig, env_index: int) -> EnvFactory:
    def thunk() -> gym.Env:
        env = gym.make(config.env_id, **config.env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(config.seed + env_index)
        if getattr(env, "observation_space", None) is not None:
            env.observation_space.seed(config.seed + env_index)
        return env

    return thunk


def make_vector_env(config: TrainConfig) -> gym.vector.SyncVectorEnv:
    env_fns = [make_env(config, env_index) for env_index in range(config.num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)
