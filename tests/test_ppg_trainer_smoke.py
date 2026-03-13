from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.ppg_trainer import train_ppg


def test_train_ppg_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppg",
        env_id="CartPole-v1",
        seed=141,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (32, 32),
            "aux_frequency": 1,
            "aux_epochs": 1,
            "aux_minibatch_size": 32,
            "aux_buffer_rollouts": 2,
        },
    )

    result = train_ppg(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
