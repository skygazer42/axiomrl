from pathlib import Path

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.rebrac_trainer import train_rebrac
from axiomrl.runtime.workflows import evaluate_checkpoint


def test_train_rebrac_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=123,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 128,
            "dataset_seed": 22,
            "batch_size": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "actor_bc_weight": 1.0,
            "critic_bc_weight": 1.0,
            "actor_q_weight": 1.0,
            "eval_interval": 8,
        },
    )

    result = train_rebrac(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 8
    assert "eval_return_mean" in result.metrics
    assert "eval_return_mean" in metrics
