from pathlib import Path

from rl_training.cli import load_config, main
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.ppo_trainer import train_ppo


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 1",
                "total_timesteps: 128",
                f"output_dir: {tmp_path}",
                "num_envs: 2",
                "algo_kwargs:",
                "  num_steps: 32",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.algo == "ppo"
    assert config.env_id == "CartPole-v1"
    assert config.seed == 1
    assert config.output_dir == tmp_path
    assert config.algo_kwargs["num_steps"] == 32


def test_train_command_runs_with_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 5",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'base-runs'}",
                "num_envs: 1",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  num_steps: 32",
                "  update_epochs: 1",
                "  minibatch_size: 32",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )

    override_dir = tmp_path / "override-runs"
    exit_code = main(
        [
            "train",
            "--config",
            str(config_file),
            "--output-dir",
            str(override_dir),
            "--total-timesteps",
            "64",
        ]
    )

    assert exit_code == 0
    assert any(override_dir.iterdir())


def test_eval_command_runs_from_checkpoint(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=23,
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
    result = train_ppo(config, run_suffix="cli-eval")

    exit_code = main(
        [
            "eval",
            "--checkpoint",
            str(result.checkpoint_path),
            "--num-episodes",
            "1",
        ]
    )

    assert exit_code == 0


def test_resume_command_runs_from_checkpoint(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=29,
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
    result = train_dqn(config, run_suffix="cli-resume")

    exit_code = main(
        [
            "resume",
            "--checkpoint",
            str(result.checkpoint_path),
            "--total-timesteps",
            "160",
        ]
    )

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]

    assert exit_code == 0
    assert len(run_dirs) >= 2


def test_train_command_runs_for_sac_config(tmp_path: Path) -> None:
    config_file = tmp_path / "sac-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: sac",
                "env_id: Pendulum-v1",
                "seed: 41",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'sac-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  hidden_sizes: [32, 32]",
                "  alpha: 0.2",
                "  tau: 0.005",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "sac-runs").iterdir())
