from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_training.api import CURL, DRQN, R2D2, DrQ, DrQv2
from rl_training.experiment.config import TrainConfig
from tests.support.public_api import register_tiny_render_env


def test_drqn_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqn",
        env_id="CartPole-v1",
        seed=60,
        total_timesteps=64,
        output_dir=tmp_path / "drqn-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "target_update_interval": 8,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 8,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "eval_interval": 16,
        },
    )

    algo = DRQN(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drqn.pt")
    loaded = DRQN.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_r2d2_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="r2d2",
        env_id="CartPole-v1",
        seed=61,
        total_timesteps=64,
        output_dir=tmp_path / "r2d2-runs",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 4,
            "learning_starts": 4,
            "train_frequency": 1,
            "target_update_interval": 8,
            "hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
            "sequence_length": 8,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
            "prioritized_beta_end": 1.0,
            "prioritized_beta_fraction": 1.0,
            "priority_eta": 0.9,
            "n_step": 3,
            "eval_interval": 16,
        },
    )

    algo = R2D2(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "r2d2.pt")
    loaded = R2D2.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_drqv2_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drqv2",
        env_id=register_tiny_render_env(),
        seed=54,
        total_timesteps=96,
        output_dir=tmp_path / "drqv2-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "tau": 0.01,
            "policy_delay": 2,
            "augmentation_pad": 4,
            "exploration_noise": 0.1,
            "exploration_noise_clip": 0.3,
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

    algo = DrQv2(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drqv2.pt")
    loaded = DrQv2.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_drq_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="drq",
        env_id=register_tiny_render_env(),
        seed=55,
        total_timesteps=96,
        output_dir=tmp_path / "drq-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "alpha": 0.1,
            "tau": 0.01,
            "augmentation_pad": 4,
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

    algo = DrQ(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "drq.pt")
    loaded = DrQ.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_curl_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="curl",
        env_id=register_tiny_render_env(),
        seed=57,
        total_timesteps=96,
        output_dir=tmp_path / "curl-runs",
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "features_dim": 64,
            "actor_hidden_sizes": (32,),
            "critic_hidden_sizes": (32,),
            "projection_dim": 32,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "alpha": 0.1,
            "tau": 0.01,
            "augmentation_pad": 4,
            "curl_temperature": 0.1,
            "curl_coef": 1.0,
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

    algo = CURL(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "curl.pt")
    loaded = CURL.load(exported)
    action = loaded.predict(np.zeros((9, 84, 84), dtype=np.uint8))
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert len(action) == 1
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}
