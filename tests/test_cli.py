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


def test_train_command_runs_for_double_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "double-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: double_dqn",
                "env_id: CartPole-v1",
                "seed: 43",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'double-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "double-dqn-runs").iterdir())


def test_train_command_runs_for_dueling_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "dueling-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: dueling_dqn",
                "env_id: CartPole-v1",
                "seed: 47",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'dueling-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "dueling-dqn-runs").iterdir())


def test_train_command_runs_for_ddpg_config(tmp_path: Path) -> None:
    config_file = tmp_path / "ddpg-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ddpg",
                "env_id: Pendulum-v1",
                "seed: 49",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'ddpg-runs'}",
                "eval_episodes: 0",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 16",
                "  learning_starts: 16",
                "  train_frequency: 1",
                "  hidden_sizes: [16, 16]",
                "  tau: 0.005",
                "  exploration_noise: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "ddpg-runs").iterdir())


def test_train_command_runs_for_td3_config(tmp_path: Path) -> None:
    config_file = tmp_path / "td3-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: td3",
                "env_id: Pendulum-v1",
                "seed: 81",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'td3-runs'}",
                "eval_episodes: 0",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 16",
                "  learning_starts: 16",
                "  train_frequency: 1",
                "  hidden_sizes: [16, 16]",
                "  tau: 0.005",
                "  exploration_noise: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "td3-runs").iterdir())


def test_train_command_runs_for_noisy_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "noisy-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: noisy_dqn",
                "env_id: CartPole-v1",
                "seed: 51",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'noisy-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "noisy-dqn-runs").iterdir())


def test_train_command_runs_for_prioritized_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "prioritized-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: prioritized_dqn",
                "env_id: CartPole-v1",
                "seed: 53",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'prioritized-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  prioritized_alpha: 0.6",
                "  prioritized_beta_start: 0.4",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "prioritized-dqn-runs").iterdir())


def test_train_command_runs_for_n_step_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "n-step-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: n_step_dqn",
                "env_id: CartPole-v1",
                "seed: 65",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'n-step-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  gamma: 0.99",
                "  n_step: 3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "n-step-dqn-runs").iterdir())


def test_train_command_runs_for_qr_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "qr-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: qr_dqn",
                "env_id: CartPole-v1",
                "seed: 67",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'qr-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  gamma: 0.99",
                "  num_quantiles: 51",
                "  kappa: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "qr-dqn-runs").iterdir())


def test_train_command_runs_for_rainbow_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "rainbow-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: rainbow_dqn",
                "env_id: CartPole-v1",
                "seed: 55",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'rainbow-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  gamma: 0.99",
                "  n_step: 3",
                "  epsilon_start: 0.0",
                "  epsilon_end: 0.0",
                "  exploration_fraction: 0.0",
                "  prioritized_alpha: 0.6",
                "  prioritized_beta_start: 0.4",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "rainbow-dqn-runs").iterdir())


def test_train_command_runs_for_c51_dqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "c51-dqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: c51_dqn",
                "env_id: CartPole-v1",
                "seed: 57",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'c51-dqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  v_min: 0.0",
                "  v_max: 200.0",
                "  num_atoms: 51",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "c51-dqn-runs").iterdir())
