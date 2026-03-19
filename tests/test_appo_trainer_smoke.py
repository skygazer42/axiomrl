from pathlib import Path

import pytest

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.appo_trainer import train_appo
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_appo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=281,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "rho_clip": 1.0,
            "c_clip": 1.0,
            "pg_rho_clip": 1.0,
        },
    )

    result = train_appo(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics


def test_train_appo_supports_local_async_backend(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=283,
        total_timesteps=64,
        output_dir=tmp_path,
        execution_backend="local_async",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "rho_clip": 1.0,
            "c_clip": 1.0,
            "pg_rho_clip": 1.0,
        },
    )

    result = train_appo(config, run_suffix="async-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics


def test_train_appo_supports_clip_coefficient_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="appo",
        env_id="CartPole-v1",
        seed=282,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "rho_clip": 1.0,
            "c_clip": 1.0,
            "pg_rho_clip": 1.0,
            "clip_coef_schedule": {
                "type": "linear",
                "start": 0.25,
                "end": 0.05,
            },
        },
    )

    result = train_appo(config, run_suffix="clip-schedule-smoke")

    assert result.checkpoint_path is not None
    assert result.metrics["clip_coef"] == pytest.approx(0.05, rel=1e-6)
