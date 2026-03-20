from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_training import IQL, PPO, TrainConfig
from rl_training.data import export_random_transition_dataset


def test_ppo_real_env_training_save_load_and_inference_cycle(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=71,
        total_timesteps=64,
        output_dir=tmp_path / "ppo-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    algo = PPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "ppo.pt")
    loaded = PPO.load(exported)

    env = gym.make("CartPole-v1")
    try:
        obs, _ = env.reset(seed=19)
        action = loaded.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert isinstance(reward, float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert np.asarray(next_obs).shape == np.asarray(obs).shape
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_iql_real_npz_dataset_training_save_load_and_inference_cycle(tmp_path: Path) -> None:
    dataset_path = export_random_transition_dataset(
        "Pendulum-v1",
        tmp_path / "datasets" / "pendulum_rollout.npz",
        num_steps=128,
        seed=23,
    )
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=72,
        total_timesteps=96,
        output_dir=tmp_path / "iql-runs",
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "npz",
            "dataset_path": str(dataset_path),
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "expectile": 0.7,
            "beta": 3.0,
            "max_advantage_weight": 100.0,
        },
    )

    algo = IQL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "iql.pt")
    loaded = IQL.load(exported)

    env = gym.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=29)
        action = loaded.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert isinstance(reward, float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert np.asarray(next_obs).shape == np.asarray(obs).shape
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}
