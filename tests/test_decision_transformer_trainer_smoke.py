from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.decision_transformer_trainer import train_decision_transformer
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_decision_transformer_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="decision_transformer",
        env_id="Pendulum-v1",
        seed=171,
        total_timesteps=8,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 17,
            "batch_size": 8,
            "context_length": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "target_return": 0.0,
            "max_timestep": 64,
        },
    )

    result = train_decision_transformer(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 8
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
