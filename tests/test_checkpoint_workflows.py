from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.ppo_trainer import train_ppo
from rl_training.runtime.sac_trainer import train_sac
from rl_training.runtime.workflows import evaluate_checkpoint, resume_training


def test_evaluate_checkpoint_returns_metrics_for_ppo(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    train_result = train_ppo(config, run_suffix="eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_resume_training_advances_global_step_for_dqn(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 256,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (16, 16),
        },
    )

    train_result = train_dqn(config, run_suffix="resume-source")
    resumed = resume_training(
        train_result.checkpoint_path,
        total_timesteps=160,
        run_suffix="resume-target",
    )

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160


def test_evaluate_checkpoint_returns_metrics_for_sac(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="sac",
        env_id="Pendulum-v1",
        seed=37,
        total_timesteps=96,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "alpha": 0.2,
            "tau": 0.005,
        },
    )

    train_result = train_sac(config, run_suffix="sac-eval-source")
    metrics = evaluate_checkpoint(train_result.checkpoint_path, num_episodes=1)

    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}
