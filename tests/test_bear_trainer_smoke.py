from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.bear_trainer import train_bear


def test_train_bear_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=107,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 256,
            "dataset_seed": 43,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "latent_dim": 2,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "behavior_kl_weight": 0.5,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
            "eval_interval": 32,
        },
    )

    result = train_bear(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
