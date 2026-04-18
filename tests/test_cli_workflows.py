import json
from pathlib import Path

import pytest

from rl_training.cli import main
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.ppo_trainer import train_ppo


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


def test_train_command_overrides_execution_backend(tmp_path: Path) -> None:
    config_file = tmp_path / "ppo-config.yaml"
    run_root = tmp_path / "runs"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 17",
                "total_timesteps: 64",
                f"output_dir: {run_root}",
                "execution_backend: local_sync",
                "num_envs: 2",
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

    exit_code = main(
        [
            "train",
            "--config",
            str(config_file),
            "--execution-backend",
            "local_async",
        ]
    )

    run_dir = next(path for path in run_root.iterdir() if path.is_dir())
    config_payload = json.loads((run_dir / "config.yaml").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert config_payload["execution_backend"] == "local_async"


def test_train_command_runs_seed_sweep_and_writes_benchmark_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "ppo-seed-sweep.yaml"
    run_root = tmp_path / "runs"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {run_root}",
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

    exit_code = main(
        [
            "train",
            "--config",
            str(config_file),
            "--seeds",
            "11,13",
        ]
    )

    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    summary_path = run_root / "benchmark-summary.json"

    assert exit_code == 0
    assert len(run_dirs) == 2
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "aggregate_metrics" in summary_payload
    assert "runs" in summary_payload
    assert {entry["seed"] for entry in summary_payload["runs"]} == {11, 13}
    captured = capsys.readouterr().out
    assert f"benchmark_summary_path={summary_path}" in captured
    assert "metrics=" in captured


def test_train_command_rejects_missing_algo_with_cli_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = tmp_path / "missing-algo.yaml"
    config_file.write_text(
        "\n".join(
            [
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'runs'}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config_file)])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: config file" in stderr
    assert "must define 'algo' or reference another config via 'config'" in stderr


def test_train_command_rejects_missing_env_id_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "missing-env-id.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'runs'}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config_file)])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: config file" in stderr
    assert "missing required key 'env_id'" in stderr


def test_train_command_rejects_missing_config_file_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "does-not-exist.yaml"

    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config_file)])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: config file" in stderr
    assert "does not exist" in stderr


def test_train_command_rejects_malformed_seed_list_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "ppo-bad-seeds.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'runs'}",
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

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train",
                "--config",
                str(config_file),
                "--seeds",
                "11,,13",
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: --seeds expects a comma-separated list of integers" in stderr


def test_train_command_rejects_negative_seed_values_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "ppo-negative-seeds.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'runs'}",
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

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train",
                "--config",
                str(config_file),
                "--seeds",
                "-1,2",
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: --seeds expects a comma-separated list of non-negative integers" in stderr


def test_train_command_reports_summary_collision_as_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "benchmark-summary.json").write_text('{"status": "existing"}', encoding="utf-8")
    config_file = tmp_path / "ppo-collision.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {run_root}",
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

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train",
                "--config",
                str(config_file),
                "--seeds",
                "11,13",
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: benchmark summary already exists" in stderr


def test_train_command_merges_seed_override_with_existing_benchmark_keys(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    config_file = tmp_path / "ppo-benchmark-merge.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {run_root}",
                "num_envs: 1",
                "eval_episodes: 1",
                "benchmark:",
                "  best_metric_mode: min",
                "  best_metric: eval_return_mean",
                "algo_kwargs:",
                "  num_steps: 32",
                "  update_epochs: 1",
                "  minibatch_size: 32",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file), "--seeds", "11,13"])

    run_dir = next(path for path in run_root.iterdir() if path.is_dir())
    payload = json.loads((run_dir / "config.yaml").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["benchmark"]["best_metric_mode"] == "min"
    assert payload["benchmark"]["best_metric"] == "eval_return_mean"
    assert "seeds" not in payload["benchmark"]


def test_train_command_rejects_invalid_benchmark_seeds_from_config_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "ppo-invalid-benchmark-seeds.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 31",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'runs'}",
                "num_envs: 1",
                "eval_episodes: 1",
                "benchmark:",
                "  seeds: not-a-sequence",
                "algo_kwargs:",
                "  num_steps: 32",
                "  update_epochs: 1",
                "  minibatch_size: 32",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config_file)])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "error: benchmark['seeds'] must be a sequence of integers" in stderr


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


def test_resume_command_can_use_compatible_config_override(tmp_path: Path) -> None:
    base_run_root = tmp_path / "base-runs"
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=29,
        total_timesteps=96,
        output_dir=base_run_root,
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
    result = train_dqn(config, run_suffix="cli-resume-config")

    resume_run_root = tmp_path / "resume-runs"
    resume_config = tmp_path / "resume-config.yaml"
    resume_config.write_text(
        "\n".join(
            [
                "algo: dqn",
                "env_id: CartPole-v1",
                "seed: 29",
                "total_timesteps: 160",
                f"output_dir: {resume_run_root}",
                "num_envs: 1",
                "eval_episodes: 2",
                "checkpoint_interval: 8",
                "env_kwargs:",
                "  render_mode: rgb_array",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "resume",
            "--checkpoint",
            str(result.checkpoint_path),
            "--config",
            str(resume_config),
        ]
    )

    run_dirs = [path for path in resume_run_root.iterdir() if path.is_dir()]

    assert exit_code == 0
    assert len(run_dirs) == 1
    config_payload = json.loads((run_dirs[0] / "config.yaml").read_text(encoding="utf-8"))
    assert config_payload["output_dir"] == str(resume_run_root)
    assert config_payload["total_timesteps"] == 160
    assert config_payload["eval_episodes"] == 2
    assert config_payload["env_kwargs"]["render_mode"] == "rgb_array"


def test_resume_command_rejects_incompatible_config_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_run_root = tmp_path / "base-runs"
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=29,
        total_timesteps=96,
        output_dir=base_run_root,
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
    result = train_dqn(config, run_suffix="cli-resume-config-bad")

    bad_config = tmp_path / "resume-config-bad.yaml"
    bad_config.write_text(
        "\n".join(
            [
                "algo: dqn",
                "env_id: MountainCar-v0",
                "seed: 29",
                "total_timesteps: 160",
                f"output_dir: {tmp_path / 'resume-runs-bad'}",
                "num_envs: 1",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [16, 16]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "resume",
                "--checkpoint",
                str(result.checkpoint_path),
                "--config",
                str(bad_config),
            ]
        )

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "resume config" in stderr
    assert "env_id='MountainCar-v0' expected 'CartPole-v1'" in stderr


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


def test_train_command_runs_for_tqc_config(tmp_path: Path) -> None:
    config_file = tmp_path / "tqc-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: tqc",
                "env_id: Pendulum-v1",
                "seed: 42",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'tqc-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  hidden_sizes: [32, 32]",
                "  alpha: 0.2",
                "  tau: 0.005",
                "  num_critics: 3",
                "  num_quantiles: 7",
                "  top_quantiles_to_drop_per_net: 1",
                "  kappa: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "tqc-runs").iterdir())


def test_train_command_runs_for_redq_config(tmp_path: Path) -> None:
    config_file = tmp_path / "redq-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: redq",
                "env_id: Pendulum-v1",
                "seed: 43",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'redq-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 256",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  gradient_updates_per_step: 2",
                "  hidden_sizes: [32, 32]",
                "  alpha: 0.2",
                "  tau: 0.005",
                "  num_critics: 5",
                "  subset_size: 2",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "redq-runs").iterdir())


def test_train_command_runs_for_iql_config(tmp_path: Path) -> None:
    config_file = tmp_path / "iql-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: iql",
                "env_id: Pendulum-v1",
                "seed: 44",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'iql-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 13",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  tau: 0.005",
                "  expectile: 0.7",
                "  beta: 3.0",
                "  max_advantage_weight: 100.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "iql-runs").iterdir())


def test_train_command_runs_for_bc_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bc-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: bc",
                "env_id: Pendulum-v1",
                "seed: 46",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'bc-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 64",
                "  dataset_seed: 19",
                "  batch_size: 16",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  eval_interval: 8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "bc-runs").iterdir())


def test_train_command_runs_for_awac_config(tmp_path: Path) -> None:
    config_file = tmp_path / "awac-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: awac",
                "env_id: Pendulum-v1",
                "seed: 73",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'awac-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 64",
                "  dataset_seed: 13",
                "  batch_size: 16",
                "  hidden_sizes: [16, 16]",
                "  eval_interval: 8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "awac-runs").iterdir())


def test_train_command_runs_for_awr_config(tmp_path: Path) -> None:
    config_file = tmp_path / "awr-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: awr",
                "env_id: Pendulum-v1",
                "seed: 74",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'awr-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 64",
                "  dataset_seed: 14",
                "  batch_size: 16",
                "  hidden_sizes: [16, 16]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  beta: 1.0",
                "  max_weight: 20.0",
                "  eval_interval: 8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "awr-runs").iterdir())


def test_train_command_runs_for_marwil_config(tmp_path: Path) -> None:
    config_file = tmp_path / "marwil-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: marwil",
                "env_id: Pendulum-v1",
                "seed: 74",
                "total_timesteps: 16",
                f"output_dir: {tmp_path / 'marwil-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 64",
                "  dataset_seed: 14",
                "  batch_size: 16",
                "  hidden_sizes: [16, 16]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  beta: 1.0",
                "  vf_coeff: 1.0",
                "  moving_average_sqd_adv_norm_start: 100.0",
                "  moving_average_sqd_adv_norm_update_rate: 0.05",
                "  eval_interval: 8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "marwil-runs").iterdir())


def test_train_command_runs_for_her_config(tmp_path: Path) -> None:
    config_file = tmp_path / "her-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: her",
                "env_id: RL-PointGoal1D-v0",
                "seed: 75",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'her-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 16",
                "  learning_starts: 8",
                "  hidden_sizes: [16, 16]",
                "  eval_interval: 8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "her-runs").iterdir())


def test_train_command_runs_for_cql_config(tmp_path: Path) -> None:
    config_file = tmp_path / "cql-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: cql",
                "env_id: Pendulum-v1",
                "seed: 45",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'cql-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 19",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  alpha: 0.2",
                "  tau: 0.005",
                "  cql_alpha: 5.0",
                "  num_cql_samples: 10",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "cql-runs").iterdir())


def test_train_command_runs_for_cal_ql_config(tmp_path: Path) -> None:
    config_file = tmp_path / "cal-ql-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: cal_ql",
                "env_id: Pendulum-v1",
                "seed: 46",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'cal-ql-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 20",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  alpha: 0.2",
                "  tau: 0.005",
                "  cql_alpha: 5.0",
                "  num_cql_samples: 10",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "cal-ql-runs").iterdir())


def test_train_command_runs_for_edac_config(tmp_path: Path) -> None:
    config_file = tmp_path / "edac-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: edac",
                "env_id: Pendulum-v1",
                "seed: 47",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'edac-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 20",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  alpha: 0.2",
                "  tau: 0.005",
                "  num_critics: 4",
                "  eta: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "edac-runs").iterdir())


def test_train_command_runs_for_rlpd_config(tmp_path: Path) -> None:
    config_file = tmp_path / "rlpd-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: rlpd",
                "env_id: Pendulum-v1",
                "seed: 48",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'rlpd-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 21",
                "  buffer_capacity: 256",
                "  batch_size: 32",
                "  learning_starts: 16",
                "  train_frequency: 1",
                "  gradient_updates_per_step: 2",
                "  offline_pretrain_updates: 4",
                "  offline_batch_ratio: 0.5",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  alpha: 0.2",
                "  tau: 0.005",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "rlpd-runs").iterdir())


def test_train_command_runs_for_xql_config(tmp_path: Path) -> None:
    config_file = tmp_path / "xql-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: xql",
                "env_id: Pendulum-v1",
                "seed: 47",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'xql-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 21",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  tau: 0.005",
                "  beta: 3.0",
                "  loss_temperature: 1.0",
                "  max_advantage_weight: 100.0",
                "  max_value_diff_exp: 5.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "xql-runs").iterdir())


def test_train_command_runs_for_bear_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bear-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: bear",
                "env_id: Pendulum-v1",
                "seed: 46",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'bear-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 20",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  latent_dim: 2",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  tau: 0.005",
                "  behavior_kl_weight: 0.5",
                "  mmd_sigma: 20.0",
                "  mmd_alpha: 10.0",
                "  num_mmd_action_samples: 10",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "bear-runs").iterdir())


def test_train_command_runs_for_bcq_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bcq-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: bcq",
                "env_id: Pendulum-v1",
                "seed: 47",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'bcq-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 21",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  latent_dim: 2",
                "  num_action_samples: 10",
                "  perturbation_scale: 0.05",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  tau: 0.005",
                "  vae_kl_weight: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "bcq-runs").iterdir())


def test_train_command_runs_for_trpo_config(tmp_path: Path) -> None:
    config_file = tmp_path / "trpo-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: trpo",
                "env_id: CartPole-v1",
                "seed: 48",
                "total_timesteps: 64",
                f"output_dir: {tmp_path / 'trpo-runs'}",
                "num_envs: 2",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  num_steps: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.001",
                "  value_updates: 3",
                "  max_kl: 0.01",
                "  cg_iterations: 5",
                "  cg_damping: 0.1",
                "  line_search_steps: 5",
                "  line_search_shrink: 0.8",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "trpo-runs").iterdir())


def test_train_command_runs_for_discrete_sac_config(tmp_path: Path) -> None:
    config_file = tmp_path / "discrete-sac-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: discrete_sac",
                "env_id: CartPole-v1",
                "seed: 49",
                "total_timesteps: 128",
                f"output_dir: {tmp_path / 'discrete-sac-runs'}",
                "num_envs: 2",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  alpha: 0.2",
                "  tau: 0.005",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "discrete-sac-runs").iterdir())


def test_train_command_runs_for_crossq_config(tmp_path: Path) -> None:
    config_file = tmp_path / "crossq-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: crossq",
                "env_id: Pendulum-v1",
                "seed: 50",
                "total_timesteps: 128",
                f"output_dir: {tmp_path / 'crossq-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  hidden_sizes: [32, 32]",
                "  critic_hidden_sizes: [32, 32]",
                "  learning_rate: 0.001",
                "  gamma: 0.99",
                "  alpha: 0.1",
                "  policy_delay: 1",
                "  adam_beta1: 0.5",
                "  bn_momentum: 0.99",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "crossq-runs").iterdir())


def test_train_command_runs_for_td3_bc_config(tmp_path: Path) -> None:
    config_file = tmp_path / "td3-bc-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: td3_bc",
                "env_id: Pendulum-v1",
                "seed: 45",
                "total_timesteps: 32",
                f"output_dir: {tmp_path / 'td3-bc-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  dataset_kind: random",
                "  dataset_size: 128",
                "  dataset_seed: 19",
                "  batch_size: 32",
                "  hidden_sizes: [32, 32]",
                "  learning_rate: 0.0003",
                "  gamma: 0.99",
                "  tau: 0.005",
                "  policy_noise: 0.2",
                "  noise_clip: 0.5",
                "  policy_delay: 2",
                "  bc_alpha: 2.5",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "td3-bc-runs").iterdir())


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
                "eval_episodes: 1",
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
                "eval_episodes: 1",
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


def test_train_command_runs_for_iqn_config(tmp_path: Path) -> None:
    config_file = tmp_path / "iqn-config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: iqn",
                "env_id: CartPole-v1",
                "seed: 69",
                "total_timesteps: 96",
                f"output_dir: {tmp_path / 'iqn-runs'}",
                "eval_episodes: 1",
                "algo_kwargs:",
                "  buffer_capacity: 512",
                "  batch_size: 32",
                "  learning_starts: 32",
                "  train_frequency: 1",
                "  target_update_interval: 16",
                "  hidden_sizes: [32, 32]",
                "  gamma: 0.99",
                "  num_quantiles: 16",
                "  embedding_dim: 32",
                "  kappa: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_file)])

    assert exit_code == 0
    assert any((tmp_path / "iqn-runs").iterdir())


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
