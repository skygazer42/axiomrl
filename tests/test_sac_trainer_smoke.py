from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.sac_trainer import train_sac


def test_train_sac_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="sac",
        env_id="Pendulum-v1",
        seed=31,
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
        },
    )

    result = train_sac(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
