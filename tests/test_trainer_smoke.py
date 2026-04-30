from pathlib import Path

import pytest

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.ppo_trainer import train_ppo


def test_train_ppo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=3,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=2,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
        },
    )

    result = train_ppo(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_ppo_supports_entropy_coefficient_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "ent_coef_schedule": {
                "type": "linear",
                "start": 0.02,
                "end": 0.001,
            },
        },
    )

    result = train_ppo(config, run_suffix="entropy-schedule-smoke")

    assert result.checkpoint_path is not None
    assert result.metrics["ent_coef"] == pytest.approx(0.001, rel=1e-6)


def test_train_ppo_supports_clip_coefficient_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=23,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "clip_coef_schedule": {
                "type": "linear",
                "start": 0.3,
                "end": 0.05,
            },
        },
    )

    result = train_ppo(config, run_suffix="clip-schedule-smoke")

    assert result.checkpoint_path is not None
    assert result.metrics["clip_coef"] == pytest.approx(0.05, rel=1e-6)


def test_train_ppo_supports_local_async_execution_backend(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=31,
        total_timesteps=128,
        output_dir=tmp_path,
        execution_backend="local_async",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
        },
    )

    result = train_ppo(config, run_suffix="async-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
