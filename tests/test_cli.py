from pathlib import Path

import pytest

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


def test_load_config_resolves_linked_zoo_preset(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "ppo.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 3",
                "total_timesteps: 128",
                f"output_dir: {tmp_path / 'runs'}",
                "num_envs: 2",
                "algo_kwargs:",
                "  num_steps: 32",
            ]
        ),
        encoding="utf-8",
    )

    preset_dir = tmp_path / "zoo"
    preset_dir.mkdir()
    preset_file = preset_dir / "cartpole.yaml"
    preset_file.write_text(
        "\n".join(
            [
                "name: cartpole_ppo",
                "config: configs/ppo.yaml",
                "algorithm: ppo",
                "env_id: CartPole-v1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(preset_file)

    assert config.algo == "ppo"
    assert config.env_id == "CartPole-v1"
    assert config.seed == 3
    assert config.output_dir == tmp_path / "runs"
    assert config.algo_kwargs["num_steps"] == 32


def test_load_config_can_resolve_packaged_repo_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/ppo/cartpole.yaml")

    assert config.algo == "ppo"
    assert config.env_id == "CartPole-v1"
    assert config.total_timesteps > 0


def test_load_config_can_resolve_packaged_ars_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/ars/pendulum.yaml")

    assert config.algo == "ars"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["num_directions"] == 8
    assert config.algo_kwargs["step_size"] == pytest.approx(0.02)


def test_load_config_can_resolve_packaged_openai_es_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/openai_es/pendulum.yaml")

    assert config.algo == "openai_es"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["num_directions"] == 8
    assert config.algo_kwargs["noise_std"] == pytest.approx(0.03)


def test_load_config_can_resolve_packaged_drqv2_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/drqv2/pendulum_pixels.yaml")

    assert config.algo == "drqv2"
    assert config.env_id == "Pendulum-v1"
    assert config.env_kwargs["render_mode"] == "rgb_array"
    assert config.env_kwargs["wrappers"]["pixels"]["frame_stack"] == 3


def test_load_config_can_resolve_packaged_drq_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/drq/pendulum_pixels.yaml")

    assert config.algo == "drq"
    assert config.env_id == "Pendulum-v1"
    assert config.env_kwargs["render_mode"] == "rgb_array"
    assert config.env_kwargs["wrappers"]["pixels"]["frame_stack"] == 3


def test_load_config_can_resolve_packaged_curl_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/curl/pendulum_pixels.yaml")

    assert config.algo == "curl"
    assert config.env_id == "Pendulum-v1"
    assert config.env_kwargs["render_mode"] == "rgb_array"
    assert config.algo_kwargs["projection_dim"] == 128


def test_load_config_can_resolve_packaged_ppg_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/ppg/cartpole.yaml")

    assert config.algo == "ppg"
    assert config.env_id == "CartPole-v1"
    assert config.algo_kwargs["aux_frequency"] == 2
    assert config.algo_kwargs["aux_buffer_rollouts"] == 4


def test_load_config_can_resolve_packaged_decision_transformer_config_outside_repo_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/decision_transformer/pendulum.yaml")

    assert config.algo == "decision_transformer"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["context_length"] == 20
    assert config.algo_kwargs["num_layers"] == 3


def test_load_config_can_resolve_packaged_impala_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/impala/cartpole.yaml")

    assert config.algo == "impala"
    assert config.env_id == "CartPole-v1"
    assert config.algo_kwargs["num_steps"] == 128
    assert config.algo_kwargs["rho_clip"] == pytest.approx(1.0)


def test_load_config_can_resolve_packaged_appo_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/appo/cartpole.yaml")

    assert config.algo == "appo"
    assert config.env_id == "CartPole-v1"
    assert config.algo_kwargs["num_steps"] == 128
    assert config.algo_kwargs["clip_coef"] == pytest.approx(0.2)


def test_load_config_can_resolve_packaged_mopo_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/mopo/pendulum.yaml")

    assert config.algo == "mopo"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["num_ensembles"] == 5
    assert config.algo_kwargs["rollout_horizon"] == 3


def test_load_config_can_resolve_packaged_pets_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/pets/pendulum.yaml")

    assert config.algo == "pets"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["num_ensembles"] == 5
    assert config.algo_kwargs["planning_horizon"] == 5


def test_load_config_can_resolve_packaged_crr_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/crr/pendulum.yaml")

    assert config.algo == "crr"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["n_action_samples"] == 4
    assert config.algo_kwargs["weight_type"] == "exp"


def test_load_config_can_resolve_packaged_awr_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/awr/pendulum.yaml")

    assert config.algo == "awr"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["beta"] == pytest.approx(1.0)
    assert config.algo_kwargs["max_weight"] == pytest.approx(20.0)


def test_load_config_can_resolve_packaged_marwil_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/marwil/pendulum.yaml")

    assert config.algo == "marwil"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["beta"] == pytest.approx(1.0)
    assert config.algo_kwargs["vf_coeff"] == pytest.approx(1.0)


def test_load_config_can_resolve_packaged_cal_ql_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/cal_ql/pendulum.yaml")

    assert config.algo == "cal_ql"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["cql_alpha"] == pytest.approx(5.0)


def test_load_config_can_resolve_packaged_edac_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/edac/pendulum.yaml")

    assert config.algo == "edac"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["num_critics"] == 10
    assert config.algo_kwargs["eta"] == pytest.approx(1.0)


def test_load_config_can_resolve_packaged_rlpd_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/rlpd/pendulum.yaml")

    assert config.algo == "rlpd"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["offline_pretrain_updates"] == 1000
    assert config.algo_kwargs["offline_batch_ratio"] == pytest.approx(0.5)


def test_load_config_can_resolve_packaged_xql_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/xql/pendulum.yaml")

    assert config.algo == "xql"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["loss_temperature"] == pytest.approx(1.0)


def test_load_config_can_resolve_packaged_rebrac_config_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("configs/rebrac/pendulum.yaml")

    assert config.algo == "rebrac"
    assert config.env_id == "Pendulum-v1"
    assert config.algo_kwargs["actor_bc_weight"] == pytest.approx(1.0)
    assert config.algo_kwargs["critic_bc_weight"] == pytest.approx(1.0)


def test_zoo_subcommand_uses_packaged_manifest_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["zoo", "--manifest", "zoo/atari/benchmark.yaml", "--format", "commands"])

    assert exit_code == 0


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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
                "eval_episodes: 0",
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
