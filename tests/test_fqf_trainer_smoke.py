from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_fqf_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="fqf",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "learning_rate": 1e-3,
            "fraction_learning_rate": 5e-4,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.3,
            "num_quantiles": 8,
            "embedding_dim": 16,
            "kappa": 1.0,
            "entropy_coef": 1e-3,
        },
    )

    result = train_dqn(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 128
    assert "fraction_loss" in result.metrics
    assert "entropy_loss" in result.metrics
    assert "eval_return_mean" in metrics
