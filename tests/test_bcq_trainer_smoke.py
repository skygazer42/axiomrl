from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.bcq_trainer import train_bcq


def test_train_bcq_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=103,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 256,
            "dataset_seed": 41,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "vae_kl_weight": 0.5,
            "eval_interval": 32,
        },
    )

    result = train_bcq(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
