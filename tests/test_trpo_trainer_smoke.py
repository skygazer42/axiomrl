from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.trpo_trainer import train_trpo


def test_train_trpo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=107,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 1e-3,
            "value_updates": 3,
            "max_kl": 0.01,
            "cg_iterations": 5,
            "cg_damping": 0.1,
            "line_search_steps": 5,
            "line_search_shrink": 0.8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
    )

    result = train_trpo(config, run_suffix="smoke")

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics


def test_train_trpo_supports_local_async_backend(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=207,
        total_timesteps=64,
        output_dir=tmp_path,
        execution_backend="local_async",
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 1e-3,
            "value_updates": 3,
            "max_kl": 0.01,
            "cg_iterations": 5,
            "cg_damping": 0.1,
            "line_search_steps": 5,
            "line_search_shrink": 0.8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
    )

    result = train_trpo(config, run_suffix="async-smoke")

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in result.metrics
