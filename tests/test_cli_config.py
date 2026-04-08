import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rl_training.cli import load_config, main
from rl_training.version import __version__


def test_version_flag_prints_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_module_does_not_import_torch_on_import() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rl_training.cli; print('torch' in sys.modules)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_config_subcommand_prints_resolved_train_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 1",
                "total_timesteps: 128",
                f"output_dir: {tmp_path}",
                "tags:",
                "  - demo",
                "algo_kwargs:",
                "  num_steps: 32",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["config", "--config", str(config_file)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["algo"] == "ppo"
    assert payload["env_id"] == "CartPole-v1"
    assert payload["seed"] == 1
    assert payload["total_timesteps"] == 128
    assert payload["output_dir"] == str(tmp_path)
    assert payload["tags"] == ["demo"]
    assert payload["algo_kwargs"]["num_steps"] == 32


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


def test_load_config_can_resolve_packaged_tennis_training_configs_to_raiddata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    for config_path in (
        "configs/rainbow_dqn/tennis_atari.yaml",
        "configs/r2d2/tennis_atari.yaml",
        "configs/ppo/tennis_atari.yaml",
        "configs/impala/tennis_atari.yaml",
    ):
        config = load_config(config_path)
        assert config.env_id == "ALE/Tennis-v5"
        assert config.output_dir == Path("/raiddata/kdsoft/axiomrl-runs")


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
