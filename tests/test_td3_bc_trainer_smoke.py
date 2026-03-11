from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.td3_bc_trainer import train_td3_bc


def test_train_td3_bc_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="td3_bc",
        env_id="Pendulum-v1",
        seed=97,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 256,
            "dataset_seed": 19,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "bc_alpha": 2.5,
        },
    )

    result = train_td3_bc(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "eval_return_mean" in result.metrics
