from pathlib import Path

from axiomrl.envs import POINT_GOAL_ENV_ID
from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.her_trainer import train_her


def test_train_her_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="her",
        env_id=POINT_GOAL_ENV_ID,
        seed=97,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "her_ratio": 0.8,
            "exploration_noise": 0.1,
            "eval_interval": 16,
        },
    )

    result = train_her(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 16
    assert "eval_success_rate" in result.metrics
