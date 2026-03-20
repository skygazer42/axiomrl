from pathlib import Path
import json
import csv
import hashlib
import io

import pytest
import yaml

from rl_training.cli import load_config, main
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn
from rl_training.runtime.ppo_trainer import train_ppo
from rl_training.version import __version__


def test_version_flag_prints_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


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
                "execution_backend: local_async",
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
    assert config.execution_backend == "local_async"
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


def test_report_subcommand_uses_packaged_manifest_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["report", "--manifest", "zoo/atari/benchmark.yaml"])

    assert exit_code == 0


def test_zoo_subcommand_supports_report_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    first_run_dir = runs_dir / "ppo__ALE-Breakout-v5__seed9__demo"
    first_run_dir.mkdir(parents=True)
    (first_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "ppo",
                "env_id": "ALE/Breakout-v5",
                "seed": 9,
                "latest_metrics": {
                    "eval_return_mean": 21.0,
                    "eval_human_normalized_score": 12.5,
                },
                "best_checkpoint": {
                    "path": str(first_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 30.0,
                },
            }
        ),
        encoding="utf-8",
    )
    second_run_dir = runs_dir / "ppo__ALE-Breakout-v5__seed11__demo"
    second_run_dir.mkdir(parents=True)
    (second_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "ppo",
                "env_id": "ALE/Breakout-v5",
                "seed": 11,
                "latest_metrics": {
                    "eval_return_mean": 27.0,
                    "eval_human_normalized_score": 18.5,
                },
                "best_checkpoint": {
                    "path": str(second_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 35.0,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ppo__ALE-Breakout-v5__seed9__demo" in captured.out
    assert "ppo__ALE-Breakout-v5__seed11__demo" in captured.out
    assert "latest_eval_return_mean=21.0" in captured.out
    assert "aggregate algo=ppo env_id=ALE/Breakout-v5 runs=2 seeds=9,11" in captured.out
    assert "latest_eval_return_mean_mean=24.0" in captured.out
    assert "latest_eval_human_normalized_score_mean=15.5" in captured.out
    assert "best_eval_return_mean_max=35.0" in captured.out


def test_report_subcommand_supports_report_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    first_run_dir = runs_dir / "ppo__ALE-Breakout-v5__seed9__demo"
    first_run_dir.mkdir(parents=True)
    (first_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "ppo",
                "env_id": "ALE/Breakout-v5",
                "seed": 9,
                "latest_metrics": {
                    "eval_return_mean": 21.0,
                    "eval_human_normalized_score": 12.5,
                },
                "best_checkpoint": {
                    "path": str(first_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 30.0,
                },
            }
        ),
        encoding="utf-8",
    )
    second_run_dir = runs_dir / "ppo__ALE-Breakout-v5__seed11__demo"
    second_run_dir.mkdir(parents=True)
    (second_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "ppo",
                "env_id": "ALE/Breakout-v5",
                "seed": 11,
                "latest_metrics": {
                    "eval_return_mean": 27.0,
                    "eval_human_normalized_score": 18.5,
                },
                "best_checkpoint": {
                    "path": str(second_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 35.0,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "report",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--runs-dir",
            str(runs_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ppo__ALE-Breakout-v5__seed9__demo" in captured.out
    assert "ppo__ALE-Breakout-v5__seed11__demo" in captured.out
    assert "latest_eval_return_mean=21.0" in captured.out
    assert "aggregate algo=ppo env_id=ALE/Breakout-v5 runs=2 seeds=9,11" in captured.out
    assert "latest_eval_return_mean_mean=24.0" in captured.out
    assert "latest_eval_human_normalized_score_mean=15.5" in captured.out
    assert "best_eval_return_mean_max=35.0" in captured.out


def test_leaderboard_subcommand_uses_packaged_manifest_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["leaderboard", "--manifest", "zoo/atari/benchmark.yaml"])

    assert exit_code == 0


def test_leaderboard_subcommand_supports_leaderboard_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, best_return in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 30.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ppo_breakout", 11, 35.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 80.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": best_return - 5.0,
                        "eval_human_normalized_score": best_return / 2.0,
                        "best_eval_return_mean": best_return,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "leaderboard",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "leaderboard" in captured.out
    assert "rank=1" in captured.out
    assert "preset_name=dqn_breakout" in captured.out
    assert "best_eval_return_mean_max=80.0" in captured.out
    assert "ppo_breakout" not in captured.out


def test_load_config_resolves_benchmark_aware_zoo_preset(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "dqn.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: dqn",
                "env_id: ALE/Breakout-v5",
                "seed: 3",
                "total_timesteps: 128",
                f"output_dir: {tmp_path / 'runs'}",
                "tags:",
                "  - atari",
                "env_kwargs:",
                "  repeat_action_probability: 0.0",
                "  wrappers:",
                "    atari:",
                "      frame_skip: 4",
            ]
        ),
        encoding="utf-8",
    )

    zoo_dir = tmp_path / "zoo" / "atari"
    zoo_dir.mkdir(parents=True)
    benchmark_manifest = zoo_dir / "benchmark.yaml"
    benchmark_manifest.write_text(
        "\n".join(
            [
                "suite: atari",
                "protocol:",
                "  name: atari_default_v1",
                "  training:",
                "    repeat_action_probability: 0.0",
                "  evaluation:",
                "    repeat_action_probability: 0.25",
                "score_normalization:",
                "  type: human_random",
                "  source: atari_breakout_reference",
                "presets:",
                "  - name: breakout_debug",
                "    config: zoo/atari/breakout_debug.yaml",
            ]
        ),
        encoding="utf-8",
    )
    preset_file = zoo_dir / "breakout_debug.yaml"
    preset_file.write_text(
        "\n".join(
            [
                "name: breakout_debug",
                "config: configs/dqn.yaml",
                "algorithm: dqn",
                "env_id: ALE/Breakout-v5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(preset_file)

    assert config.benchmark["suite"] == "atari"
    assert config.benchmark["preset_name"] == "breakout_debug"
    assert config.benchmark["protocol_name"] == "atari_default_v1"
    assert config.benchmark["score_normalization"]["random_score"] == pytest.approx(1.7)
    assert config.benchmark["score_normalization"]["human_score"] == pytest.approx(30.5)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)


def test_zoo_subcommand_supports_csv_report_output_and_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, env_id, seed, latest_return, latest_hns, best_return in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ALE/Breakout-v5", 9, 21.0, 12.5, 30.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ALE/Breakout-v5", 11, 27.0, 18.5, 35.0),
        ("dqn__ALE-Pong-v5__seed3__demo", "dqn", "ALE/Pong-v5", 3, 15.0, 8.0, 22.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": env_id,
                    "seed": seed,
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "csv",
            "--env-id",
            "ALE/Breakout-v5",
            "--sort-by",
            "best_eval_return_mean",
            "--descending",
        ]
    )

    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))

    assert exit_code == 0
    assert [row["kind"] for row in rows] == ["run", "run", "aggregate"]
    assert rows[0]["run_id"] == "ppo__ALE-Breakout-v5__seed11__demo"
    assert rows[1]["run_id"] == "ppo__ALE-Breakout-v5__seed9__demo"
    assert rows[2]["algo"] == "ppo"
    assert rows[2]["env_id"] == "ALE/Breakout-v5"
    assert rows[2]["runs"] == "2"
    assert rows[2]["best_eval_return_mean_max"] == "35.0"


def test_zoo_subcommand_can_write_csv_report_file(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, seed, latest_return, latest_hns, best_return in [
        ("ppo__ALE-Breakout-v5__seed9__demo", 9, 21.0, 12.5, 30.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", 11, 27.0, 18.5, 35.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": "ppo",
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    output_path = tmp_path / "reports" / "benchmark_report.csv"
    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "csv",
            "--output",
            str(output_path),
        ]
    )

    rows = list(csv.DictReader(io.StringIO(output_path.read_text(encoding="utf-8"))))

    assert exit_code == 0
    assert output_path.exists()
    assert [row["kind"] for row in rows] == ["run", "run", "aggregate"]
    assert rows[0]["run_id"] == "ppo__ALE-Breakout-v5__seed11__demo"
    assert rows[2]["best_eval_return_mean_max"] == "35.0"


def test_zoo_subcommand_supports_preset_columns_and_top_k_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, best_return in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 30.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ppo_breakout", 11, 35.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 80.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": best_return - 5.0,
                        "eval_human_normalized_score": best_return / 2.0,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "csv",
            "--sort-by",
            "best_eval_return_mean",
            "--descending",
            "--top-k",
            "1",
        ]
    )

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))

    assert exit_code == 0
    assert [row["kind"] for row in rows] == ["run", "aggregate"]
    assert rows[0]["run_id"] == "dqn__ALE-Breakout-v5__seed7__demo"
    assert rows[0]["preset_name"] == "dqn_breakout"
    assert rows[0]["protocol_name"] == "atari_default_v1"


def test_zoo_subcommand_supports_leaderboard_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, best_return in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 30.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ppo_breakout", 11, 35.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 80.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": best_return - 5.0,
                        "eval_human_normalized_score": best_return / 2.0,
                        "best_eval_return_mean": best_return,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "leaderboard" in captured.out
    assert "rank=1" in captured.out
    assert "preset_name=dqn_breakout" in captured.out
    assert "best_eval_return_mean_max=80.0" in captured.out
    assert "ppo_breakout" not in captured.out


def test_zoo_subcommand_leaderboard_json_includes_manifest_protocol_and_preset_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 78.0, 50.0, 90.0, 60.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ppo_breakout", 11, 80.0, 54.0, 94.0, 64.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 70.0, 35.0, 84.0, 45.0),
        ("dqn__ALE-Breakout-v5__seed8__demo", "dqn", "dqn_breakout", 8, 72.0, 38.0, 86.0, 46.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    top_entry = payload["entries"][0]

    assert exit_code == 0
    assert payload["protocol_metadata"]["name"] == "atari_default_v1"
    assert payload["protocol_metadata"]["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)
    assert payload["score_normalization_metadata"]["source"] == "atari_breakout_reference"
    assert payload["score_normalization_metadata"]["human_score"] == pytest.approx(30.5)
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout"]
    assert top_entry["protocol_metadata"]["training"]["frameskip"] == 1
    assert top_entry["score_normalization_metadata"]["game"] == "breakout"
    assert top_entry["preset_metadata"]["config"] == "zoo/atari/ppo_breakout.yaml"
    assert top_entry["preset_metadata"]["description"] == "On-policy Atari baseline with shared CNN actor-critic."


def test_zoo_subcommand_leaderboard_json_includes_manifest_identity_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 78.0, 50.0, 90.0, 60.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo", "ppo_breakout", 11, 80.0, 54.0, 94.0, 64.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 70.0, 35.0, 84.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    manifest_path = Path("zoo/atari/benchmark.yaml")
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    expected_fingerprint = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    exit_code = main(
        [
            "zoo",
            "--manifest",
            str(manifest_path),
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_metadata"]["suite"] == "atari"
    assert payload["manifest_metadata"]["preset_count"] == len(manifest_payload["presets"])
    assert payload["manifest_metadata"]["preset_names"][:2] == ["dqn_breakout", "apex_dqn_breakout"]
    assert payload["manifest_metadata"]["fingerprint"] == expected_fingerprint
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout"]


def test_zoo_subcommand_leaderboard_json_includes_packaged_manifest_source_metadata(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", 9, 78.0, 50.0, 90.0, 60.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 70.0, 35.0, 84.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    resolved_path = Path(payload["manifest_source"]["resolved_path"])

    assert exit_code == 0
    assert payload["manifest_source"]["requested_path"] == "zoo/atari/benchmark.yaml"
    assert payload["manifest_source"]["source_kind"] == "packaged_asset"
    assert resolved_path.name == "benchmark.yaml"
    assert resolved_path.exists()
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout"]


def test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 78.0, 50.0, 90.0, 60.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_legacy_v0", 7, 70.0, 35.0, 84.0, 45.0),
        ("ghost__ALE-Breakout-v5__seed5__demo", "ppo", "ghost_breakout", "atari_default_v1", 5, 68.0, 41.0, 80.0, 48.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": protocol_name,
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = payload["manifest_alignment_summary"]

    assert exit_code == 0
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout"]
    assert payload["entries"][0]["manifest_alignment_status"] == "aligned"
    assert payload["entries"][0]["manifest_alignment_severity"] == "clean"
    assert summary["total_runs"] == 3
    assert summary["aligned_runs"] == 1
    assert summary["drifted_runs"] == 2
    assert summary["unknown_preset_runs"] == 1
    assert summary["protocol_mismatch_runs"] == 1
    assert summary["all_runs_aligned"] is False
    assert summary["severity"] == "error"
    assert summary["drifted_presets"] == ["dqn_breakout", "ghost_breakout"]


def test_zoo_subcommand_leaderboard_can_fail_on_manifest_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 78.0, 50.0, 90.0, 60.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_legacy_v0", 7, 70.0, 35.0, 84.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": protocol_name,
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--fail-on-manifest-drift",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_summary"]["drifted_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "warning"
    assert payload["entries"][0]["preset_name"] == "ppo_breakout"


def test_zoo_subcommand_leaderboard_can_threshold_fail_on_manifest_drift_severity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 78.0, 50.0, 90.0, 60.0),
        ("ghost__ALE-Breakout-v5__seed7__demo", "ppo", "ghost_breakout", "atari_default_v1", 7, 70.0, 35.0, 84.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": protocol_name,
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--fail-on-manifest-drift-severity",
            "error",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_summary"]["drifted_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "error"
    assert any(entry["preset_name"] == "ghost_breakout" for entry in payload["entries"])


def test_zoo_subcommand_leaderboard_can_filter_fail_on_manifest_drift_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 78.0, 50.0, 90.0, 60.0),
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_legacy_v0", 7, 70.0, 35.0, 84.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": protocol_name,
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--fail-on-manifest-drift-type",
            "protocol-mismatch",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_summary"]["unknown_preset_runs"] == 0
    assert payload["manifest_alignment_summary"]["protocol_mismatch_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "warning"
    assert any(entry["preset_name"] == "dqn_breakout" for entry in payload["entries"])


def test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_fail_reasons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 78.0, 50.0, 90.0, 60.0),
        ("ghost__ALE-Breakout-v5__seed7__demo", "ppo", "ghost_breakout", "atari_default_v1", 7, 70.0, 35.0, 84.0, 45.0),
        ("dqn__ALE-Breakout-v5__seed5__demo", "dqn", "dqn_breakout", "atari_legacy_v0", 5, 68.0, 41.0, 80.0, 48.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": protocol_name,
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--fail-on-manifest-drift-severity",
            "error",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_fail_reasons"] == ["unknown-preset"]
    assert payload["manifest_alignment_summary"]["unknown_preset_runs"] == 1
    assert payload["manifest_alignment_summary"]["protocol_mismatch_runs"] == 1


def test_zoo_subcommand_leaderboard_json_includes_seed_count_and_rank_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 75.0, 30.0, 80.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 70.0, 32.0, 85.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 58.0, 40.0, 60.0, 50.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sort_by"] == "best_eval_human_normalized_score"
    assert payload["descending"] is True
    assert payload["entries"][0]["preset_name"] == "ppo_breakout"
    assert payload["entries"][0]["seed_count"] == 1
    assert payload["entries"][0]["rank_best_eval_return_mean"] == 2
    assert payload["entries"][0]["rank_latest_eval_return_mean"] == 2
    assert payload["entries"][0]["rank_best_eval_human_normalized_score"] == 1
    assert payload["entries"][0]["rank_latest_eval_human_normalized_score"] == 1
    assert payload["entries"][0]["best_over_latest_eval_return_ratio"] == pytest.approx(60.0 / 58.0)
    assert payload["entries"][1]["preset_name"] == "dqn_breakout"
    assert payload["entries"][1]["seed_count"] == 2
    assert payload["entries"][1]["rank_best_eval_return_mean"] == 1
    assert payload["entries"][1]["rank_latest_eval_return_mean"] == 1
    assert payload["entries"][1]["rank_best_eval_human_normalized_score"] == 2
    assert payload["entries"][1]["rank_latest_eval_human_normalized_score"] == 2


def test_zoo_subcommand_leaderboard_supports_min_seeds_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 75.0, 30.0, 80.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 70.0, 32.0, 85.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 58.0, 40.0, 60.0, 50.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--min-seeds",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["min_seeds"] == 2
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["preset_name"] == "dqn_breakout"
    assert payload["entries"][0]["seed_count"] == 2


def test_zoo_subcommand_leaderboard_supports_latest_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 20.0, 100.0, 30.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 92.0, 22.0, 105.0, 32.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 60.0, 40.0, 70.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "latest-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "latest-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]


def test_zoo_subcommand_leaderboard_supports_gap_return_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 20.0, 100.0, 30.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 95.0, 22.0, 102.0, 32.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 40.0, 40.0, 70.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "gap-return",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "gap-return"
    assert payload["sort_by"] == "best_minus_latest_eval_return_mean_gap"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]


def test_zoo_subcommand_leaderboard_supports_compare_to_latest_on_normalized_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 20.0, 100.0, 30.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 92.0, 22.0, 105.0, 32.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 60.0, 40.0, 70.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--compare-to",
            "latest",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["compare_to"] == "latest"
    assert payload["leaderboard_metric"] == "latest-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]


def test_zoo_subcommand_leaderboard_supports_compare_to_latest_on_unnormalized_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "benchmark.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "suite: demo",
                "protocol:",
                "  name: demo_default_v1",
                "score_normalization:",
                "  type: none",
                "presets: []",
            ]
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    for run_id, algo, env_id, seed, latest_return, best_return in [
        ("dqn__DemoEnv-v1__seed7__demo", "dqn", "DemoEnv-v1", 7, 90.0, 100.0),
        ("ppo__DemoEnv-v2__seed3__demo", "ppo", "DemoEnv-v2", 3, 120.0, 130.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": env_id,
                    "seed": seed,
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "best_eval_return_mean": best_return,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            str(manifest_path),
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--compare-to",
            "latest",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["compare_to"] == "latest"
    assert payload["leaderboard_metric"] == "latest-return"
    assert payload["sort_by"] == "latest_eval_return_mean"
    assert payload["descending"] is True
    assert [entry["env_id"] for entry in payload["entries"]] == ["DemoEnv-v2", "DemoEnv-v1"]


def test_zoo_subcommand_leaderboard_supports_score_view_return_on_normalized_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 20.0, 100.0, 30.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 92.0, 22.0, 105.0, 32.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 60.0, 40.0, 70.0, 45.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--compare-to",
            "latest",
            "--score-view",
            "return",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["compare_to"] == "latest"
    assert payload["score_view"] == "return"
    assert payload["leaderboard_metric"] == "latest-return"
    assert payload["sort_by"] == "latest_eval_return_mean"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["dqn_breakout", "ppo_breakout"]


def test_zoo_subcommand_leaderboard_rejects_normalized_score_view_without_normalization(tmp_path: Path) -> None:
    manifest_path = tmp_path / "benchmark.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "suite: demo",
                "protocol:",
                "  name: demo_default_v1",
                "score_normalization:",
                "  type: none",
                "presets: []",
            ]
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "dqn__DemoEnv-v1__seed7__demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "dqn",
                "env_id": "DemoEnv-v1",
                "seed": 7,
                "latest_metrics": {
                    "eval_return_mean": 90.0,
                    "best_eval_return_mean": 100.0,
                },
                "best_checkpoint": {
                    "path": str(run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="score normalization"):
        main(
            [
                "zoo",
                "--manifest",
                str(manifest_path),
                "--format",
                "leaderboard",
                "--runs-dir",
                str(runs_dir),
                "--report-output",
                "json",
                "--score-view",
                "normalized",
            ]
        )


def test_zoo_subcommand_leaderboard_supports_stability_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 30.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 95.0, 50.0, 102.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 60.0, 40.0, 70.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 62.0, 42.0, 72.0, 46.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "stability-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "stability-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score_std"
    assert payload["descending"] is False
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]
    assert payload["entries"][0]["latest_eval_human_normalized_score_std"] == pytest.approx(1.4142135623730951)
    assert payload["entries"][1]["latest_eval_human_normalized_score_std"] == pytest.approx(14.142135623730951)


def test_zoo_subcommand_leaderboard_supports_confidence_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 90.0, 30.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 95.0, 50.0, 102.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 60.0, 40.0, 70.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 62.0, 42.0, 72.0, 46.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "confidence-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "confidence-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score_ci95"
    assert payload["descending"] is False
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]
    assert payload["entries"][0]["latest_eval_human_normalized_score_stderr"] == pytest.approx(1.0)
    assert payload["entries"][0]["latest_eval_human_normalized_score_ci95"] == pytest.approx(1.96)
    assert payload["entries"][1]["latest_eval_human_normalized_score_stderr"] == pytest.approx(10.0)
    assert payload["entries"][1]["latest_eval_human_normalized_score_ci95"] == pytest.approx(19.6)


def test_zoo_subcommand_leaderboard_supports_median_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 50.0, 20.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 60.0, 40.0, 102.0, 36.0),
        ("dqn__ALE-Breakout-v5__seed11__demo", "dqn", "dqn_breakout", 11, 100.0, 70.0, 104.0, 37.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 45.0, 80.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 72.0, 50.0, 82.0, 46.0),
        ("ppo__ALE-Breakout-v5__seed7__demo", "ppo", "ppo_breakout", 7, 74.0, 55.0, 84.0, 47.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "median-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "median-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score_median"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]
    assert payload["entries"][0]["latest_eval_human_normalized_score_median"] == pytest.approx(50.0)
    assert payload["entries"][1]["latest_eval_human_normalized_score_median"] == pytest.approx(40.0)


def test_zoo_subcommand_leaderboard_supports_iqr_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 50.0, 20.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 60.0, 40.0, 102.0, 36.0),
        ("dqn__ALE-Breakout-v5__seed11__demo", "dqn", "dqn_breakout", 11, 100.0, 70.0, 104.0, 37.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 45.0, 80.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 72.0, 50.0, 82.0, 46.0),
        ("ppo__ALE-Breakout-v5__seed7__demo", "ppo", "ppo_breakout", 7, 74.0, 55.0, 84.0, 47.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--leaderboard-metric",
            "iqr-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "iqr-normalized"
    assert payload["sort_by"] == "latest_eval_human_normalized_score_iqr"
    assert payload["descending"] is False
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout"]
    assert payload["entries"][0]["latest_eval_human_normalized_score_iqr"] == pytest.approx(5.0)
    assert payload["entries"][1]["latest_eval_human_normalized_score_iqr"] == pytest.approx(25.0)


def test_zoo_subcommand_leaderboard_supports_delta_vs_baseline_normalized_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 102.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 40.0, 80.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 74.0, 50.0, 82.0, 46.0),
        ("a2c__ALE-Breakout-v5__seed1__demo", "a2c", "a2c_breakout", 1, 45.0, 10.0, 60.0, 15.0),
        ("a2c__ALE-Breakout-v5__seed2__demo", "a2c", "a2c_breakout", 2, 47.0, 30.0, 61.0, 16.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--baseline-preset",
            "dqn_breakout",
            "--leaderboard-metric",
            "delta-vs-baseline-normalized",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["baseline_preset"] == "dqn_breakout"
    assert payload["leaderboard_metric"] == "delta-vs-baseline-normalized"
    assert payload["sort_by"] == "delta_vs_baseline_latest_eval_human_normalized_score_mean"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout", "a2c_breakout"]
    assert payload["entries"][0]["delta_vs_baseline_latest_eval_human_normalized_score_mean"] == pytest.approx(12.0)
    assert payload["entries"][1]["delta_vs_baseline_latest_eval_human_normalized_score_mean"] == pytest.approx(0.0)
    assert payload["entries"][2]["delta_vs_baseline_latest_eval_human_normalized_score_mean"] == pytest.approx(-13.0)


def test_zoo_subcommand_leaderboard_supports_ratio_vs_baseline_return_metric_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 102.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 40.0, 80.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 74.0, 50.0, 82.0, 46.0),
        ("a2c__ALE-Breakout-v5__seed1__demo", "a2c", "a2c_breakout", 1, 45.0, 10.0, 60.0, 15.0),
        ("a2c__ALE-Breakout-v5__seed2__demo", "a2c", "a2c_breakout", 2, 47.0, 30.0, 61.0, 16.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--baseline-preset",
            "dqn_breakout",
            "--leaderboard-metric",
            "ratio-vs-baseline-return",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["leaderboard_metric"] == "ratio-vs-baseline-return"
    assert payload["sort_by"] == "ratio_vs_baseline_latest_eval_return_mean_mean"
    assert payload["descending"] is True
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout", "dqn_breakout", "a2c_breakout"]
    assert payload["entries"][0]["ratio_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(72.0 / 60.0)
    assert payload["entries"][1]["ratio_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(1.0)
    assert payload["entries"][2]["ratio_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(46.0 / 60.0)


def test_zoo_subcommand_leaderboard_rejects_baseline_metric_without_baseline_preset(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "dqn__ALE-Breakout-v5__seed7__demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "dqn",
                "env_id": "ALE/Breakout-v5",
                "seed": 7,
                "benchmark": {
                    "suite": "atari",
                    "preset_name": "dqn_breakout",
                    "protocol_name": "atari_default_v1",
                },
                "latest_metrics": {
                    "eval_return_mean": 55.0,
                    "eval_human_normalized_score": 30.0,
                    "best_eval_return_mean": 80.0,
                    "best_eval_human_normalized_score": 46.0,
                },
                "best_checkpoint": {
                    "path": str(run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 80.0,
                    "eval_human_normalized_score": 46.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline-preset"):
        main(
            [
                "zoo",
                "--manifest",
                "zoo/atari/benchmark.yaml",
                "--format",
                "leaderboard",
                "--runs-dir",
                str(runs_dir),
                "--group-by",
                "preset",
                "--report-output",
                "json",
                "--leaderboard-metric",
                "delta-vs-baseline-normalized",
            ]
        )


def test_zoo_subcommand_leaderboard_includes_baseline_summary_top_movers_and_regressions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 100.0, 35.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 102.0, 36.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 40.0, 80.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 74.0, 50.0, 82.0, 46.0),
        ("ppg__ALE-Breakout-v5__seed1__demo", "ppg", "ppg_breakout", 1, 64.0, 35.0, 70.0, 41.0),
        ("ppg__ALE-Breakout-v5__seed2__demo", "ppg", "ppg_breakout", 2, 68.0, 41.0, 71.0, 42.0),
        ("a2c__ALE-Breakout-v5__seed1__demo", "a2c", "a2c_breakout", 1, 45.0, 10.0, 60.0, 15.0),
        ("a2c__ALE-Breakout-v5__seed2__demo", "a2c", "a2c_breakout", 2, 47.0, 30.0, 61.0, 16.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": algo,
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_return,
                        "eval_human_normalized_score": latest_hns,
                        "best_eval_return_mean": best_return,
                        "best_eval_human_normalized_score": best_hns,
                    },
                    "best_checkpoint": {
                        "path": str(run_dir / "checkpoints" / "best.pt"),
                        "metric_name": "eval_return_mean",
                        "metric_value": best_return,
                        "eval_human_normalized_score": best_hns,
                    },
                }
            ),
            encoding="utf-8",
        )

    exit_code = main(
        [
            "zoo",
            "--manifest",
            "zoo/atari/benchmark.yaml",
            "--format",
            "leaderboard",
            "--runs-dir",
            str(runs_dir),
            "--group-by",
            "preset",
            "--report-output",
            "json",
            "--baseline-preset",
            "dqn_breakout",
            "--leaderboard-metric",
            "delta-vs-baseline-normalized",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = payload["baseline_summary"]

    assert exit_code == 0
    assert [entry["preset_name"] for entry in payload["entries"]] == ["ppo_breakout"]
    assert [entry["preset_name"] for entry in summary["top_movers_by_normalized_delta"]] == [
        "ppo_breakout",
        "ppg_breakout",
        "a2c_breakout",
    ]
    assert [entry["preset_name"] for entry in summary["top_regressions_by_return_delta"]] == [
        "a2c_breakout",
        "ppg_breakout",
        "ppo_breakout",
    ]
    assert summary["top_movers_by_normalized_delta"][1]["delta_vs_baseline_latest_eval_human_normalized_score_mean"] == (
        pytest.approx(5.0)
    )


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


def test_train_command_rejects_missing_env_id_with_cli_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_train_command_rejects_missing_config_file_with_cli_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
