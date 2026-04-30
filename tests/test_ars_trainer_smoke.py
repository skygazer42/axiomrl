from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.ars_trainer import train_ars
from axiomrl.runtime.workflows import evaluate_checkpoint


def test_train_ars_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ars",
        env_id="Pendulum-v1",
        seed=403,
        total_timesteps=100,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "hidden_sizes": (32, 32),
            "step_size": 0.02,
            "noise_std": 0.03,
            "num_directions": 2,
            "num_top_directions": 2,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    result = train_ars(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 100
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
