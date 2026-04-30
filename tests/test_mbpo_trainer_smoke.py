from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.mbpo_trainer import train_mbpo


def test_train_mbpo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="mbpo",
        env_id="Pendulum-v1",
        seed=41,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "synthetic_buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "model_hidden_sizes": (64, 64),
            "num_ensembles": 3,
            "model_batch_size": 32,
            "model_updates": 2,
            "rollout_batch_size": 32,
            "rollout_horizon": 1,
            "rollout_refresh_interval": 32,
            "synthetic_batch_ratio": 0.5,
            "policy_learning_rate": 1e-3,
            "model_learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
        },
    )

    result = train_mbpo(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
