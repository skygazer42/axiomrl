from pathlib import Path

from rl_training.envs import POINT_GOAL_ENV_ID
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.awac_trainer import train_awac
from rl_training.runtime.bc_trainer import train_bc
from rl_training.runtime.bear_trainer import train_bear
from rl_training.runtime.bcq_trainer import train_bcq
from rl_training.runtime.her_trainer import train_her
from rl_training.runtime.controls import EarlyStoppingCallback, EarlyStoppingConfig, build_control_callbacks
from rl_training.runtime.iql_trainer import train_iql
from rl_training.runtime.trainer import TrainerState


def test_early_stopping_callback_stops_when_target_is_reached(tmp_path: Path) -> None:
    callback = EarlyStoppingCallback(
        EarlyStoppingConfig(
            metric="eval_return_mean",
            mode="max",
            patience=3,
            target_value=100.0,
        )
    )
    trainer = TrainerState(algorithm="ppo", run_dir=tmp_path)
    trainer.global_step = 32

    callback.on_eval_end(trainer, {"eval_return_mean": 120.0, "eval_episodes": 1.0})

    assert trainer.should_stop is True
    assert trainer.stop_reason is not None


def test_early_stopping_callback_stops_after_patience_is_exhausted(tmp_path: Path) -> None:
    callback = EarlyStoppingCallback(
        EarlyStoppingConfig(
            metric="eval_return_mean",
            mode="max",
            patience=1,
            min_delta=0.0,
        )
    )
    trainer = TrainerState(algorithm="ppo", run_dir=tmp_path)
    trainer.global_step = 64

    callback.on_eval_end(trainer, {"eval_return_mean": 10.0, "eval_episodes": 1.0})
    callback.on_eval_end(trainer, {"eval_return_mean": 10.0, "eval_episodes": 1.0})
    callback.on_eval_end(trainer, {"eval_return_mean": 9.0, "eval_episodes": 1.0})

    assert trainer.should_stop is True
    assert trainer.stop_reason is not None


def test_build_control_callbacks_creates_early_stopping_callback() -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=1,
        total_timesteps=64,
        output_dir=Path("runs/test"),
        algo_kwargs={
            "early_stopping": {
                "metric": "eval_return_mean",
                "mode": "max",
                "patience": 2,
            }
        },
    )

    callbacks = build_control_callbacks(config)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStoppingCallback)


def test_iql_supports_eval_interval_and_early_stopping_config(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=7,
        total_timesteps=8,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 17,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 4,
            "early_stopping": {
                "metric": "eval_return_mean",
                "mode": "max",
                "patience": 0,
                "target_value": -1.0,
            },
        },
    )

    result = train_iql(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_bc_supports_max_epochs_and_learning_rate_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bc",
        env_id="Pendulum-v1",
        seed=5,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 13,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
            "max_epochs": 4,
            "warmup_steps": 2,
            "learning_rate_schedule": {
                "type": "linear",
                "start": 1.0,
                "end": 0.25,
            },
        },
    )

    result = train_bc(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["epoch"] <= 4.0
    assert "lr_scale" in result.metrics
    assert "learning_rate" in result.metrics


def test_awac_supports_eval_interval_and_early_stopping_config(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=9,
        total_timesteps=8,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 21,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 4,
            "early_stopping": {
                "metric": "eval_return_mean",
                "mode": "max",
                "patience": 0,
                "target_value": -1.0,
            },
        },
    )

    result = train_awac(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_bcq_supports_max_updates_and_learning_rate_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=10,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 23,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "num_action_samples": 10,
            "eval_interval": 8,
            "max_updates": 5,
            "warmup_steps": 2,
            "learning_rate_schedule": "cosine",
        },
    )

    result = train_bcq(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["update_count"] <= 5.0
    assert "lr_scale" in result.metrics
    assert "learning_rate" in result.metrics


def test_bear_supports_max_updates_and_learning_rate_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=12,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 25,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "eval_interval": 8,
            "max_updates": 5,
            "warmup_steps": 2,
            "learning_rate_schedule": "cosine",
        },
    )

    result = train_bear(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["update_count"] <= 5.0
    assert "lr_scale" in result.metrics
    assert "learning_rate" in result.metrics


def test_iql_supports_max_updates_and_learning_rate_schedule(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=8,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 19,
            "batch_size": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
            "max_updates": 5,
            "warmup_steps": 2,
            "learning_rate_schedule": "cosine",
        },
    )

    result = train_iql(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["update_count"] <= 5.0
    assert "lr_scale" in result.metrics
    assert "learning_rate" in result.metrics


def test_her_supports_eval_interval_and_early_stopping_config(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=11,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 128,
            "batch_size": 8,
            "learning_starts": 4,
            "hidden_sizes": (16, 16),
            "eval_interval": 4,
            "early_stopping": {
                "metric": "eval_success_rate",
                "mode": "max",
                "patience": 0,
                "target_value": -1.0,
            },
        },
    )

    result = train_her(config, run_suffix="controls-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
