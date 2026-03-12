from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.rlpd_trainer import train_rlpd
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_rlpd_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=191,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 41,
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "gradient_updates_per_step": 2,
            "offline_pretrain_updates": 4,
            "offline_batch_ratio": 0.5,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "eval_interval": 16,
        },
    )

    result = train_rlpd(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 16
    assert result.metrics["pretrain_updates_done"] >= 4
    assert result.metrics["offline_batch_size"] >= 0
    assert result.metrics["online_batch_size"] >= 0
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics

