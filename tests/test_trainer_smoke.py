from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.ppo_trainer import train_ppo


def test_train_ppo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=3,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=2,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
        },
    )

    result = train_ppo(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
