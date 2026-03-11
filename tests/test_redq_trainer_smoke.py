from pathlib import Path

import pytest

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.redq_trainer import train_redq


def test_train_redq_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=47,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "gradient_updates_per_step": 2,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 5,
            "subset_size": 2,
        },
    )

    result = train_redq(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert result.metrics["gradient_steps"] >= 2
    assert "eval_return_mean" in result.metrics


def test_train_redq_rejects_non_positive_gradient_updates_per_step(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=47,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=0,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "gradient_updates_per_step": 0,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 5,
            "subset_size": 2,
        },
    )

    with pytest.raises(ValueError, match="gradient_updates_per_step must be >= 1"):
        train_redq(config, run_suffix="invalid-gradient-updates")
