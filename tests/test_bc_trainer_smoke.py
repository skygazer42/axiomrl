from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.bc_trainer import train_bc


def test_train_bc_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bc",
        env_id="Pendulum-v1",
        seed=83,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 11,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "eval_interval": 16,
        },
    )

    result = train_bc(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 16
    assert "eval_return_mean" in result.metrics
