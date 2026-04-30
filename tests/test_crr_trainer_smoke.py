from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.crr_trainer import train_crr
from axiomrl.runtime.workflows import evaluate_checkpoint


def test_train_crr_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=121,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 21,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
            "eval_interval": 8,
        },
    )

    result = train_crr(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 8
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
