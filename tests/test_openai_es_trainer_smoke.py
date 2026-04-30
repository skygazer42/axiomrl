from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.openai_es_trainer import train_openai_es
from axiomrl.runtime.workflows import evaluate_checkpoint


def test_train_openai_es_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=423,
        total_timesteps=100,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "hidden_sizes": (32, 32),
            "step_size": 0.02,
            "noise_std": 0.03,
            "num_directions": 2,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    result = train_openai_es(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 100
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
