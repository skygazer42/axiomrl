from pathlib import Path

import pytest

import rl_training.runtime.dqn_trainer as dqn_trainer
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=11,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
        },
    )

    result = train_dqn(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_double_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="double_dqn",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="double-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_dueling_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dueling_dqn",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="dueling-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_noisy_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="noisy_dqn",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="noisy-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_prioritized_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="prioritized_dqn",
        env_id="CartPole-v1",
        seed=23,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="prioritized-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_rainbow_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rainbow_dqn",
        env_id="CartPole-v1",
        seed=29,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "exploration_fraction": 0.0,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="rainbow-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_c51_dqn_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="c51_dqn",
        env_id="CartPole-v1",
        seed=31,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=2,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "v_min": 0.0,
            "v_max": 200.0,
            "num_atoms": 51,
        },
    )

    result = get_algorithm_spec(config.algo).train_fn(config, run_suffix="c51-smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_n_step_dqn_uses_n_step_accumulator(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="n_step_dqn",
        env_id="CartPole-v1",
        seed=37,
        total_timesteps=3,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "n_step": 3,
            "gamma": 0.99,
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "exploration_fraction": 0.0,
        },
    )

    result = train_dqn(config, run_suffix="n-step-smoke")

    assert result.metrics["global_step"] >= 3
    assert result.metrics["buffer_size"] < result.metrics["global_step"]


def test_train_rainbow_dqn_uses_n_step_accumulator_and_effective_gamma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_kwargs: dict[str, float] = {}
    original_build_algorithm = dqn_trainer._build_algorithm

    def spy_build_algorithm(*args, **kwargs):  # type: ignore[no-untyped-def]
        build_kwargs["gamma"] = float(kwargs["gamma"])
        return original_build_algorithm(*args, **kwargs)

    monkeypatch.setattr(dqn_trainer, "_build_algorithm", spy_build_algorithm)

    gamma = 0.99
    n_step = 3
    config = TrainConfig(
        algo="rainbow_dqn",
        env_id="MountainCar-v0",
        seed=43,
        total_timesteps=2,
        output_dir=tmp_path,
        eval_episodes=0,
        algo_kwargs={
            "n_step": n_step,
            "gamma": gamma,
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "exploration_fraction": 0.0,
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
        },
    )

    result = train_dqn(config, run_suffix="rainbow-n-step-smoke")

    assert build_kwargs["gamma"] == pytest.approx(gamma**n_step)
    assert result.metrics["global_step"] >= 2
    assert result.metrics["buffer_size"] == pytest.approx(0.0)


def test_train_qr_dqn_checkpoint_can_be_evaluated(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="qr_dqn",
        env_id="CartPole-v1",
        seed=41,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "num_quantiles": 51,
            "kappa": 1.0,
        },
    )

    result = train_dqn(config, run_suffix="qr-smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.checkpoint_path is not None
    assert "eval_return_mean" in metrics


def test_train_iqn_checkpoint_can_be_evaluated(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iqn",
        env_id="CartPole-v1",
        seed=47,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "num_quantiles": 16,
            "embedding_dim": 32,
            "kappa": 1.0,
        },
    )

    result = train_dqn(config, run_suffix="iqn-smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.checkpoint_path is not None
    assert "eval_return_mean" in metrics
