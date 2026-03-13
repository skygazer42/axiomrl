from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.d4pg_trainer import train_d4pg


def test_train_d4pg_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="d4pg",
        env_id="Pendulum-v1",
        seed=107,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "exploration_noise": 0.1,
            "v_min": -50.0,
            "v_max": 10.0,
            "num_atoms": 21,
        },
    )

    result = train_d4pg(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
