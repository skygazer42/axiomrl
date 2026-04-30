from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.tqc_trainer import train_tqc


def test_train_tqc_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="tqc",
        env_id="Pendulum-v1",
        seed=41,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 3,
            "num_quantiles": 7,
            "top_quantiles_to_drop_per_net": 1,
            "kappa": 1.0,
        },
    )

    result = train_tqc(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics


def test_train_tqc_supports_local_async_backend(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="tqc",
        env_id="Pendulum-v1",
        seed=141,
        total_timesteps=128,
        output_dir=tmp_path,
        execution_backend="local_async",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 3,
            "num_quantiles": 7,
            "top_quantiles_to_drop_per_net": 1,
            "kappa": 1.0,
        },
    )

    result = train_tqc(config, run_suffix="async-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
