from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training.envs import (
    POINT_GOAL_ENV_ID,
    build_env,
    flatten_goal_observation,
    goal_env_compute_done,
    goal_env_compute_reward,
    register_builtin_goal_envs,
)
from rl_training.experiment.config import TrainConfig


def test_builtin_goal_env_is_registered_and_buildable(tmp_path: Path) -> None:
    register_builtin_goal_envs()
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=5,
        total_timesteps=32,
        output_dir=tmp_path,
    )

    env = build_env(config, env_index=0)
    obs, _ = env.reset(seed=config.seed)

    assert set(obs) >= {"observation", "achieved_goal", "desired_goal"}
    assert env.action_space.shape == (1,)

    env.close()


def test_flatten_goal_observation_supports_single_and_batched_payloads() -> None:
    single = {
        "observation": np.asarray([0.25], dtype=np.float32),
        "achieved_goal": np.asarray([0.25], dtype=np.float32),
        "desired_goal": np.asarray([0.75], dtype=np.float32),
    }
    batched = {
        "observation": np.asarray([[0.25], [0.5]], dtype=np.float32),
        "achieved_goal": np.asarray([[0.25], [0.5]], dtype=np.float32),
        "desired_goal": np.asarray([[0.75], [0.1]], dtype=np.float32),
    }

    flat_single = flatten_goal_observation(single)
    flat_batch = flatten_goal_observation(batched)

    assert flat_single.shape == (2,)
    assert flat_batch.shape == (2, 2)


def test_goal_reward_and_done_helpers_support_success_and_preserved_truncation() -> None:
    register_builtin_goal_envs()
    env = gym.make(POINT_GOAL_ENV_ID)

    success_reward = goal_env_compute_reward(
        env,
        achieved_goal=np.asarray([0.5], dtype=np.float32),
        desired_goal=np.asarray([0.5], dtype=np.float32),
        info={},
    )
    success_done = goal_env_compute_done(
        env,
        achieved_goal=np.asarray([0.5], dtype=np.float32),
        desired_goal=np.asarray([0.5], dtype=np.float32),
        info={},
    )
    preserved_timeout_done = goal_env_compute_done(
        env,
        achieved_goal=np.asarray([0.0], dtype=np.float32),
        desired_goal=np.asarray([1.0], dtype=np.float32),
        info={},
        fallback_truncated=True,
    )

    assert success_reward == 0.0
    assert success_done == 1.0
    assert preserved_timeout_done == 1.0

    env.close()
