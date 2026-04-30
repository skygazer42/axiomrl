from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.cal_ql_trainer import train_cal_ql
from axiomrl.runtime.workflows import evaluate_checkpoint


def test_train_cal_ql_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=133,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 27,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "num_cql_samples": 10,
            "eval_interval": 8,
        },
    )

    result = train_cal_ql(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 8
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
