from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn


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
