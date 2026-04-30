from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.awr_trainer import train_awr


def test_train_awr_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=151,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 256,
            "dataset_seed": 33,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "returns_to_go_gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
            "normalize_advantages": True,
        },
    )

    result = train_awr(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "returns_to_go_mean" in result.metrics
    assert "eval_return_mean" in result.metrics
