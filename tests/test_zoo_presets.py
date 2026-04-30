from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiomrl.cli import load_config
from axiomrl.zoo_cli import main as zoo_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_atari_benchmark_manifest_points_to_existing_configs() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert payload["score_normalization"]["type"] == "human_random"
    presets = payload["presets"]
    assert isinstance(presets, list)
    assert presets

    for preset in presets:
        config_path = REPO_ROOT / preset["config"]
        assert config_path.exists()


def test_each_atari_zoo_preset_resolves_to_a_train_config() -> None:
    preset_paths = []
    for path in sorted((REPO_ROOT / "zoo" / "atari").glob("*.yaml")):
        if path.name.endswith("benchmark.yaml"):
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and ("algo" in payload or "config" in payload):
            preset_paths.append(path)

    assert preset_paths
    assert (REPO_ROOT / "zoo" / "atari" / "horizon_imagination_breakout.yaml").exists()
    assert (REPO_ROOT / "zoo" / "atari" / "po_dreamer_breakout.yaml").exists()
    assert (REPO_ROOT / "zoo" / "atari" / "twisted_breakout.yaml").exists()

    expected_algorithms = {
        "a2c",
        "agent57",
        "apex_dqn",
        "c51_dqn",
        "double_dqn",
        "dqn",
        "diamond",
        "horizon_imagination",
        "po_dreamer",
        "twisted",
        "eadream",
        "dreamerv3",
        "efficientzero",
        "scalezero",
        "dueling_dqn",
        "fqf",
        "gumbel_muzero",
        "impala",
        "iqn",
        "jowa",
        "mow",
        "muzero",
        "n_step_dqn",
        "noisy_dqn",
        "ppg",
        "ppo",
        "prioritized_dqn",
        "qr_dqn",
        "rainbow_dqn",
        "r2d2",
        "recurrent_ppo",
        "spr",
    }
    for preset_path in preset_paths:
        config = load_config(preset_path)
        assert config.algo in expected_algorithms
        assert config.env_id in {"ALE/Breakout-v5", "ALE/Tennis-v5"}
        assert "atari" in config.tags


def test_packaged_zoo_preset_can_be_loaded_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/dqn_breakout.yaml")

    assert config.algo == "dqn"
    assert config.env_id == "ALE/Breakout-v5"
    assert "atari" in config.tags


def test_packaged_zoo_preset_inherits_manifest_benchmark_and_protocol_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/dqn_breakout.yaml")

    assert config.benchmark["suite"] == "atari"
    assert config.benchmark["preset_name"] == "dqn_breakout"
    assert config.benchmark["protocol_name"] == "atari_default_v1"
    score_normalization = config.benchmark["score_normalization"]
    assert score_normalization["source"] == "atari_breakout_reference"
    assert score_normalization["random_score"] == pytest.approx(1.7)
    assert score_normalization["human_score"] == pytest.approx(30.5)
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)


def test_tennis_benchmark_manifest_lists_expected_presets() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "tennis_benchmark.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert [preset["name"] for preset in payload["presets"]] == [
        "rainbow_dqn_tennis",
        "r2d2_tennis",
        "apex_dqn_tennis",
        "agent57_tennis",
        "efficientzero_tennis",
    ]


def test_tennis_tuning_stage1_manifest_lists_expected_presets() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "tennis_tuning_stage1.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert [preset["name"] for preset in payload["presets"]] == [
        "apex_dqn_tennis_stable_lr",
        "apex_dqn_tennis_explore_tuned",
        "apex_dqn_tennis_reward_lite",
        "rainbow_dqn_tennis_stable_lr",
        "rainbow_dqn_tennis_no_early_stop",
        "rainbow_dqn_tennis_reward_lite",
    ]


def test_tennis_focus_manifest_lists_expected_presets() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "tennis_focus.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert [preset["name"] for preset in payload["presets"]] == [
        "apex_dqn_tennis_stable_lr",
        "apex_dqn_tennis_event_shaped",
        "rainbow_dqn_tennis_no_early_stop",
        "rainbow_dqn_tennis_event_shaped",
    ]


def test_tennis_focus_v2_manifest_lists_expected_presets() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "tennis_focus_v2.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert [preset["name"] for preset in payload["presets"]] == [
        "apex_dqn_tennis_stable_lr",
        "apex_dqn_tennis_event_v2",
        "rainbow_dqn_tennis_no_early_stop",
        "rainbow_dqn_tennis_event_v2",
    ]


def test_tennis_offense_focus_manifest_lists_expected_presets() -> None:
    manifest_path = REPO_ROOT / "zoo" / "atari" / "tennis_offense_focus.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "atari"
    assert payload["protocol"]["name"] == "atari_default_v1"
    assert [preset["name"] for preset in payload["presets"]] == [
        "apex_dqn_tennis_event_shaped",
        "apex_dqn_tennis_event_offense",
        "rainbow_dqn_tennis_event_shaped",
        "rainbow_dqn_tennis_event_offense",
    ]


@pytest.mark.parametrize(
    ("preset_name", "expected_algo"),
    [
        ("rainbow_dqn_tennis", "rainbow_dqn"),
        ("apex_dqn_tennis", "apex_dqn"),
        ("agent57_tennis", "agent57"),
        ("efficientzero_tennis", "efficientzero"),
    ],
)
def test_packaged_tennis_zoo_preset_inherits_manifest_protocol_defaults(
    monkeypatch, tmp_path: Path, preset_name: str, expected_algo: str
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(f"zoo/atari/{preset_name}.yaml")

    assert config.algo == expected_algo
    assert config.env_id == "ALE/Tennis-v5"
    assert config.benchmark["suite"] == "atari"
    assert config.benchmark["preset_name"] == preset_name
    assert config.benchmark["protocol_name"] == "atari_default_v1"
    assert "score_normalization" not in config.benchmark
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("preset_name", "expected_algo"),
    [
        ("apex_dqn_tennis_stable_lr", "apex_dqn"),
        ("apex_dqn_tennis_explore_tuned", "apex_dqn"),
        ("apex_dqn_tennis_reward_lite", "apex_dqn"),
        ("rainbow_dqn_tennis_stable_lr", "rainbow_dqn"),
        ("rainbow_dqn_tennis_no_early_stop", "rainbow_dqn"),
        ("rainbow_dqn_tennis_reward_lite", "rainbow_dqn"),
    ],
)
def test_packaged_tennis_tuning_stage1_preset_inherits_manifest_protocol_defaults(
    monkeypatch, tmp_path: Path, preset_name: str, expected_algo: str
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(f"zoo/atari/{preset_name}.yaml")

    assert config.algo == expected_algo
    assert config.env_id == "ALE/Tennis-v5"
    assert config.benchmark["suite"] == "atari"
    assert config.benchmark["preset_name"] == preset_name
    assert config.benchmark["protocol_name"] == "atari_default_v1"
    assert "score_normalization" not in config.benchmark
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("preset_name", "expected_algo"),
    [
        ("apex_dqn_tennis_event_shaped", "apex_dqn"),
        ("rainbow_dqn_tennis_event_shaped", "rainbow_dqn"),
        ("apex_dqn_tennis_event_offense", "apex_dqn"),
        ("rainbow_dqn_tennis_event_offense", "rainbow_dqn"),
        ("apex_dqn_tennis_event_v2", "apex_dqn"),
        ("rainbow_dqn_tennis_event_v2", "rainbow_dqn"),
    ],
)
def test_packaged_tennis_event_shaped_preset_inherits_manifest_protocol_defaults(
    monkeypatch, tmp_path: Path, preset_name: str, expected_algo: str
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(f"zoo/atari/{preset_name}.yaml")

    assert config.algo == expected_algo
    assert config.env_id == "ALE/Tennis-v5"
    assert config.benchmark["suite"] == "atari"
    assert config.benchmark["preset_name"] == preset_name
    assert config.benchmark["protocol_name"] == "atari_default_v1"
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)
    assert "tennis_events" in config.env_kwargs["training"]["wrappers"]


@pytest.mark.parametrize(
    ("preset_name", "expected_algo"),
    [
        ("apex_dqn_tennis_event_v2", "apex_dqn"),
        ("rainbow_dqn_tennis_event_v2", "rainbow_dqn"),
    ],
)
def test_packaged_tennis_event_v2_preset_has_stop_loss_and_v2_wrapper_defaults(
    monkeypatch, tmp_path: Path, preset_name: str, expected_algo: str
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(f"zoo/atari/{preset_name}.yaml")

    assert config.algo == expected_algo
    assert config.total_timesteps == 100000000
    early_stopping = config.algo_kwargs["early_stopping"]
    assert early_stopping["metric"] == "eval_return_mean"
    assert early_stopping["mode"] == "max"
    assert early_stopping["min_steps"] == 20000000
    assert early_stopping["patience"] == 12
    assert early_stopping["min_delta"] == pytest.approx(0.5)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["min_cross_delta_x_px"] == pytest.approx(6.0)
    assert tennis_events["cross_cooldown_steps"] == 2
    assert tennis_events["max_step_shaping_abs"] == pytest.approx(0.25)
    assert tennis_events["emit_info_metrics"] is True


def test_packaged_tennis_event_v5_preset_has_outcome_anchoring_and_fast_eval(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_v5.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["point_win_bonus"] == pytest.approx(0.5)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.5)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.0)


def test_packaged_tennis_event_v5_1_preset_restores_v4_dense_shaping_with_mild_win_anchor(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_v5_1.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.06)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.4)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.04)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.0)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.005)


def test_packaged_tennis_event_offense_v2_preset_preserves_v5_1_stability_and_adds_offense_pressure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_offense_v2.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.06)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.4)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.005)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.06)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.015)
    assert tennis_events["wide_landing_bonus"] == pytest.approx(0.01)


def test_packaged_tennis_event_offense_v3_preset_shifts_from_rally_control_toward_winning_points(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_offense_v3.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.05)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.4)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.008)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.12)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.0)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.025)
    assert tennis_events["wide_landing_bonus"] == pytest.approx(0.015)
    assert tennis_events["max_step_shaping_abs"] == pytest.approx(0.1)


def test_packaged_tennis_event_offense_v4_preset_separates_point_wins_from_generic_returns(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_offense_v4.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.03)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.08)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.01)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.16)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.02)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.035)
    assert tennis_events["wide_landing_bonus"] == pytest.approx(0.025)
    assert tennis_events["max_step_shaping_abs"] == pytest.approx(0.16)


def test_packaged_tennis_event_offense_v5_preset_rewards_attack_conversion_over_neutral_rallies(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_offense_v5.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.0)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.02)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.005)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.14)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.03)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.005)
    assert tennis_events["wide_landing_bonus"] == pytest.approx(0.005)
    assert tennis_events["attack_window_steps"] == 12
    assert tennis_events["attack_conversion_bonus"] == pytest.approx(0.1)
    assert tennis_events["failed_attack_penalty"] == pytest.approx(0.03)
    assert tennis_events["max_step_shaping_abs"] == pytest.approx(0.24)


def test_packaged_tennis_event_offense_v5_1_preset_keeps_attack_conversion_but_reduces_instability_pressure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config("zoo/atari/apex_dqn_tennis_event_offense_v5_1.yaml")

    assert config.algo == "apex_dqn"
    assert config.total_timesteps == 100000000
    assert int(config.algo_kwargs["eval_interval"]) == 100000
    assert config.env_kwargs["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert config.env_kwargs["evaluation"]["repeat_action_probability"] == pytest.approx(0.0)
    tennis_events = config.env_kwargs["training"]["wrappers"]["tennis_events"]
    assert tennis_events["successful_return_bonus"] == pytest.approx(0.01)
    assert tennis_events["failure_penalty"] == pytest.approx(-0.01)
    assert tennis_events["net_cross_bonus"] == pytest.approx(0.005)
    assert tennis_events["point_win_bonus"] == pytest.approx(0.1)
    assert tennis_events["point_loss_penalty"] == pytest.approx(0.02)
    assert tennis_events["deep_landing_bonus"] == pytest.approx(0.004)
    assert tennis_events["wide_landing_bonus"] == pytest.approx(0.004)
    assert tennis_events["attack_window_steps"] == 12
    assert tennis_events["attack_conversion_bonus"] == pytest.approx(0.05)
    assert tennis_events["failed_attack_penalty"] == pytest.approx(0.01)
    assert tennis_events["max_step_shaping_abs"] == pytest.approx(0.16)


def test_zoo_cli_can_resolve_packaged_manifest_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = zoo_main(["--manifest", "zoo/atari/benchmark.yaml", "--format", "commands"])

    assert exit_code == 0


def test_zoo_cli_report_summarizes_run_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    first_run_dir = runs_dir / "dqn__ALE-Breakout-v5__seed7__demo"
    first_run_dir.mkdir(parents=True)
    (first_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "dqn",
                "env_id": "ALE/Breakout-v5",
                "seed": 7,
                "latest_metrics": {
                    "eval_return_mean": 55.0,
                    "eval_human_normalized_score": 33.0,
                    "best_eval_return_mean": 80.0,
                },
                "best_checkpoint": {
                    "path": str(first_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 80.0,
                },
            }
        ),
        encoding="utf-8",
    )
    second_run_dir = runs_dir / "dqn__ALE-Breakout-v5__seed9__demo"
    second_run_dir.mkdir(parents=True)
    (second_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "dqn",
                "env_id": "ALE/Breakout-v5",
                "seed": 9,
                "latest_metrics": {
                    "eval_return_mean": 65.0,
                    "eval_human_normalized_score": 45.0,
                    "best_eval_return_mean": 95.0,
                },
                "best_checkpoint": {
                    "path": str(second_run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 95.0,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "suite=atari" in captured.out
    assert "protocol=atari_default_v1" in captured.out
    assert "score_normalization=human_random" in captured.out
    assert "dqn__ALE-Breakout-v5__seed7__demo" in captured.out
    assert "dqn__ALE-Breakout-v5__seed9__demo" in captured.out
    assert "latest_eval_return_mean=55.0" in captured.out
    assert "latest_eval_human_normalized_score=33.0" in captured.out
    assert "best_eval_return_mean=80.0" in captured.out
    assert "aggregate algo=dqn env_id=ALE/Breakout-v5 runs=2 seeds=7,9" in captured.out
    assert "latest_eval_return_mean_mean=60.0" in captured.out
    assert "latest_eval_human_normalized_score_mean=39.0" in captured.out
    assert "best_eval_return_mean_max=95.0" in captured.out


def test_zoo_cli_report_can_emit_filtered_sorted_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, env_id, seed, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "ALE/Breakout-v5", 7, 33.0, 80.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "ALE/Breakout-v5", 9, 45.0, 95.0),
        ("ppo__ALE-Pong-v5__seed5__demo", "ppo", "ALE/Pong-v5", 5, 17.0, 40.0),
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
                        "eval_return_mean": latest_hns + 10.0,
                        "eval_human_normalized_score": latest_hns,
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--algo",
            "dqn",
            "--sort-by",
            "latest_eval_human_normalized_score",
            "--descending",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["suite"] == "atari"
    assert payload["protocol"] == "atari_default_v1"
    assert payload["score_normalization"] == "human_random"
    assert payload["runs_dir"] == str(runs_dir)
    assert [run["run_id"] for run in payload["runs"]] == [
        "dqn__ALE-Breakout-v5__seed9__demo",
        "dqn__ALE-Breakout-v5__seed7__demo",
    ]
    assert len(payload["aggregates"]) == 1
    assert payload["aggregates"][0]["algo"] == "dqn"
    assert payload["aggregates"][0]["env_id"] == "ALE/Breakout-v5"
    assert payload["aggregates"][0]["best_eval_return_mean_max"] == pytest.approx(95.0)


def test_zoo_cli_report_json_includes_manifest_protocol_and_preset_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 33.0, 80.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 45.0, 95.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 72.0, 48.0, 102.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    dqn_run = payload["runs"][0]
    dqn_aggregate = next(aggregate for aggregate in payload["aggregates"] if aggregate["preset_name"] == "dqn_breakout")

    assert exit_code == 0
    assert payload["protocol_metadata"]["name"] == "atari_default_v1"
    assert payload["protocol_metadata"]["training"]["repeat_action_probability"] == pytest.approx(0.0)
    assert payload["protocol_metadata"]["evaluation"]["repeat_action_probability"] == pytest.approx(0.25)
    assert payload["score_normalization_metadata"]["type"] == "human_random"
    assert payload["score_normalization_metadata"]["source"] == "atari_breakout_reference"
    assert payload["score_normalization_metadata"]["random_score"] == pytest.approx(1.7)
    assert payload["score_normalization_metadata"]["human_score"] == pytest.approx(30.5)
    assert dqn_run["preset_metadata"]["config"] == "zoo/atari/dqn_breakout.yaml"
    assert dqn_run["preset_metadata"]["description"] == "Value-based Atari baseline with NatureCNN features."
    assert dqn_aggregate["protocol_metadata"]["description"].startswith("Deterministic ALE training")
    assert dqn_aggregate["score_normalization_metadata"]["game"] == "breakout"
    assert dqn_aggregate["preset_metadata"]["config"] == "zoo/atari/dqn_breakout.yaml"
    assert dqn_aggregate["preset_metadata"]["description"] == "Value-based Atari baseline with NatureCNN features."


def test_zoo_cli_report_json_includes_manifest_identity_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
                    "eval_human_normalized_score": 33.0,
                    "best_eval_return_mean": 80.0,
                },
                "best_checkpoint": {
                    "path": str(run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 80.0,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_payload = yaml.safe_load((REPO_ROOT / "zoo" / "atari" / "benchmark.yaml").read_text(encoding="utf-8"))
    expected_fingerprint = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_metadata"]["suite"] == "atari"
    assert payload["manifest_metadata"]["preset_count"] == len(manifest_payload["presets"])
    assert payload["manifest_metadata"]["preset_names"][0] == "dqn_breakout"
    assert payload["manifest_metadata"]["preset_names"][-1] == "recurrent_ppo_breakout"
    assert payload["manifest_metadata"]["fingerprint"] == expected_fingerprint


def test_zoo_cli_report_json_includes_manifest_source_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
                    "eval_human_normalized_score": 33.0,
                    "best_eval_return_mean": 80.0,
                },
                "best_checkpoint": {
                    "path": str(run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 80.0,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"

    exit_code = zoo_main(
        [
            "--manifest",
            str(manifest_path),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_source"]["requested_path"] == str(manifest_path)
    assert payload["manifest_source"]["resolved_path"] == str(manifest_path)
    assert payload["manifest_source"]["source_kind"] == "filesystem"


def test_zoo_cli_report_json_includes_manifest_alignment_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_default_v1", 7, 55.0, 33.0, 80.0),
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_legacy_v0", 9, 72.0, 48.0, 92.0),
        ("ghost__ALE-Breakout-v5__seed5__demo", "ppo", "ghost_breakout", "atari_default_v1", 5, 61.0, 40.0, 78.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    runs_by_id = {run["run_id"]: run for run in payload["runs"]}
    aggregates_by_preset = {aggregate["group"]: aggregate for aggregate in payload["aggregates"]}
    summary = payload["manifest_alignment_summary"]

    assert exit_code == 0
    assert runs_by_id["dqn__ALE-Breakout-v5__seed7__demo"]["manifest_alignment_status"] == "aligned"
    assert runs_by_id["ppo__ALE-Breakout-v5__seed9__demo"]["manifest_alignment_status"] == "protocol_mismatch"
    assert runs_by_id["ghost__ALE-Breakout-v5__seed5__demo"]["manifest_alignment_status"] == "preset_unknown"
    assert runs_by_id["dqn__ALE-Breakout-v5__seed7__demo"]["manifest_alignment_severity"] == "clean"
    assert runs_by_id["ppo__ALE-Breakout-v5__seed9__demo"]["manifest_alignment_severity"] == "warning"
    assert runs_by_id["ghost__ALE-Breakout-v5__seed5__demo"]["manifest_alignment_severity"] == "error"
    assert aggregates_by_preset["dqn_breakout"]["manifest_alignment_status"] == "aligned"
    assert aggregates_by_preset["dqn_breakout"]["manifest_alignment_severity"] == "clean"
    assert aggregates_by_preset["ppo_breakout"]["manifest_alignment_severity"] == "warning"
    assert aggregates_by_preset["ghost_breakout"]["manifest_alignment_severity"] == "error"
    assert aggregates_by_preset["ppo_breakout"]["manifest_alignment_protocol_mismatch_runs"] == 1
    assert aggregates_by_preset["ghost_breakout"]["manifest_alignment_unknown_preset_runs"] == 1
    assert summary["total_runs"] == 3
    assert summary["aligned_runs"] == 1
    assert summary["drifted_runs"] == 2
    assert summary["unknown_preset_runs"] == 1
    assert summary["protocol_mismatch_runs"] == 1
    assert summary["all_runs_aligned"] is False
    assert summary["severity"] == "error"
    assert summary["drifted_presets"] == ["ghost_breakout", "ppo_breakout"]


def test_zoo_cli_report_can_fail_on_manifest_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_default_v1", 7, 55.0, 33.0, 80.0),
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_legacy_v0", 9, 72.0, 48.0, 92.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--fail-on-manifest-drift",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_summary"]["drifted_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "warning"
    assert any(run["run_id"] == "ppo__ALE-Breakout-v5__seed9__demo" for run in payload["runs"])


def test_zoo_cli_report_can_threshold_fail_on_manifest_drift_severity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_default_v1", 7, 55.0, 33.0, 80.0),
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_legacy_v0", 9, 72.0, 48.0, 92.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--fail-on-manifest-drift-severity",
            "error",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_alignment_summary"]["drifted_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "warning"


def test_zoo_cli_report_can_filter_fail_on_manifest_drift_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_default_v1", 7, 55.0, 33.0, 80.0),
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_legacy_v0", 9, 72.0, 48.0, 92.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--fail-on-manifest-drift-type",
            "unknown-preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_alignment_summary"]["unknown_preset_runs"] == 0
    assert payload["manifest_alignment_summary"]["protocol_mismatch_runs"] == 1
    assert payload["manifest_alignment_summary"]["severity"] == "warning"


def test_zoo_cli_report_json_includes_manifest_alignment_fail_reasons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, protocol_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", "atari_legacy_v0", 7, 55.0, 33.0, 80.0),
        ("ppo__ALE-Breakout-v5__seed9__demo", "ppo", "ppo_breakout", "atari_default_v1", 9, 72.0, 48.0, 92.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--fail-on-manifest-drift-type",
            "protocol-mismatch",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["manifest_alignment_fail_reasons"] == ["protocol-mismatch"]
    assert payload["manifest_alignment_summary"]["protocol_mismatch_runs"] == 1


def test_zoo_cli_report_can_write_json_output_file(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "dqn__ALE-Breakout-v5__seed7__demo"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "dqn",
                "env_id": "ALE/Breakout-v5",
                "seed": 7,
                "latest_metrics": {
                    "eval_return_mean": 55.0,
                    "eval_human_normalized_score": 33.0,
                    "best_eval_return_mean": 80.0,
                },
                "best_checkpoint": {
                    "path": str(run_dir / "checkpoints" / "best.pt"),
                    "metric_name": "eval_return_mean",
                    "metric_value": 80.0,
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "benchmark_report.json"

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert output_path.exists()
    assert payload["suite"] == "atari"
    assert payload["runs"][0]["run_id"] == "dqn__ALE-Breakout-v5__seed7__demo"


def test_zoo_cli_report_supports_preset_grouping_and_top_k_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, preset_name, seed, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn_breakout", 7, 33.0, 80.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn_breakout", 9, 45.0, 95.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo_breakout", 5, 51.0, 120.0),
        ("ppo__ALE-Breakout-v5__seed11__demo", "ppo_breakout", 11, 58.0, 140.0),
    ]:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "algo": "ppo" if preset_name == "ppo_breakout" else "dqn",
                    "env_id": "ALE/Breakout-v5",
                    "seed": seed,
                    "benchmark": {
                        "suite": "atari",
                        "preset_name": preset_name,
                        "protocol_name": "atari_default_v1",
                    },
                    "latest_metrics": {
                        "eval_return_mean": latest_hns + 10.0,
                        "eval_human_normalized_score": latest_hns,
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
            "--sort-by",
            "best_eval_return_mean",
            "--descending",
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["group_by"] == "preset"
    assert payload["top_k"] == 1
    assert payload["runs"][0]["preset_name"] == "ppo_breakout"
    assert payload["runs"][0]["protocol_name"] == "atari_default_v1"
    assert len(payload["runs"]) == 1
    assert len(payload["aggregates"]) == 1
    assert payload["aggregates"][0]["preset_name"] == "ppo_breakout"
    assert payload["aggregates"][0]["runs"] == 2


def test_zoo_cli_report_json_includes_latest_vs_best_deltas(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
                    "eval_human_normalized_score": 33.0,
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["runs"][0]["best_minus_latest_eval_return_mean"] == pytest.approx(25.0)
    assert payload["runs"][0]["best_minus_latest_eval_human_normalized_score"] == pytest.approx(13.0)
    assert payload["aggregates"][0]["best_minus_latest_eval_return_mean_gap"] == pytest.approx(25.0)
    assert payload["aggregates"][0]["best_minus_latest_eval_human_normalized_score_gap"] == pytest.approx(13.0)


def test_zoo_cli_report_json_aggregates_include_seed_count_ratios_and_ranks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0, 44.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 50.0, 40.0, 70.0, 45.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
            "--sort-by",
            "best_eval_return_mean",
            "--descending",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    aggregates_by_preset = {aggregate["preset_name"]: aggregate for aggregate in payload["aggregates"]}

    assert exit_code == 0
    assert payload["aggregates"][0]["preset_name"] == "dqn_breakout"
    assert aggregates_by_preset["dqn_breakout"]["seed_count"] == 2
    assert aggregates_by_preset["dqn_breakout"]["best_over_latest_eval_return_ratio"] == pytest.approx(80.0 / 60.0)
    assert aggregates_by_preset["dqn_breakout"]["best_over_latest_eval_human_normalized_score_ratio"] == pytest.approx(
        46.0 / 33.0
    )
    assert aggregates_by_preset["dqn_breakout"]["rank_best_eval_return_mean"] == 1
    assert aggregates_by_preset["dqn_breakout"]["rank_latest_eval_return_mean"] == 1
    assert aggregates_by_preset["dqn_breakout"]["rank_best_eval_human_normalized_score"] == 1
    assert aggregates_by_preset["dqn_breakout"]["rank_latest_eval_human_normalized_score"] == 2
    assert aggregates_by_preset["ppo_breakout"]["seed_count"] == 1
    assert aggregates_by_preset["ppo_breakout"]["rank_best_eval_return_mean"] == 2
    assert aggregates_by_preset["ppo_breakout"]["rank_latest_eval_return_mean"] == 2
    assert aggregates_by_preset["ppo_breakout"]["rank_best_eval_human_normalized_score"] == 2
    assert aggregates_by_preset["ppo_breakout"]["rank_latest_eval_human_normalized_score"] == 1


def test_zoo_cli_report_json_aggregates_include_latest_stability_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0, 44.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    aggregate = payload["aggregates"][0]

    assert exit_code == 0
    assert aggregate["latest_eval_return_mean_min"] == pytest.approx(55.0)
    assert aggregate["latest_eval_return_mean_max"] == pytest.approx(65.0)
    assert aggregate["latest_eval_return_mean_std"] == pytest.approx(7.0710678118654755)
    assert aggregate["latest_eval_human_normalized_score_min"] == pytest.approx(30.0)
    assert aggregate["latest_eval_human_normalized_score_max"] == pytest.approx(36.0)
    assert aggregate["latest_eval_human_normalized_score_std"] == pytest.approx(4.242640687119285)


def test_zoo_cli_report_json_aggregates_include_latest_confidence_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0, 44.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    aggregate = payload["aggregates"][0]

    assert exit_code == 0
    assert aggregate["latest_eval_return_mean_stderr"] == pytest.approx(5.0)
    assert aggregate["latest_eval_return_mean_ci95"] == pytest.approx(9.8)
    assert aggregate["latest_eval_human_normalized_score_stderr"] == pytest.approx(3.0)
    assert aggregate["latest_eval_human_normalized_score_ci95"] == pytest.approx(5.88)


def test_zoo_cli_report_json_aggregates_include_latest_robustness_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 50.0, 20.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 60.0, 40.0, 78.0, 44.0),
        ("dqn__ALE-Breakout-v5__seed11__demo", "dqn", "dqn_breakout", 11, 100.0, 70.0, 81.0, 47.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    aggregate = payload["aggregates"][0]

    assert exit_code == 0
    assert aggregate["latest_eval_return_mean_median"] == pytest.approx(60.0)
    assert aggregate["latest_eval_return_mean_iqr"] == pytest.approx(25.0)
    assert aggregate["latest_eval_human_normalized_score_median"] == pytest.approx(40.0)
    assert aggregate["latest_eval_human_normalized_score_iqr"] == pytest.approx(25.0)


def test_zoo_cli_report_json_aggregates_include_baseline_comparison_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0, 44.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 40.0, 72.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 74.0, 50.0, 74.0, 46.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
            "--baseline-preset",
            "dqn_breakout",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    aggregates_by_preset = {aggregate["preset_name"]: aggregate for aggregate in payload["aggregates"]}

    assert exit_code == 0
    assert payload["baseline_preset"] == "dqn_breakout"
    assert aggregates_by_preset["dqn_breakout"]["delta_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(0.0)
    assert aggregates_by_preset["dqn_breakout"]["ratio_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(1.0)
    assert aggregates_by_preset["ppo_breakout"]["baseline_latest_eval_return_mean_mean"] == pytest.approx(60.0)
    assert aggregates_by_preset["ppo_breakout"]["baseline_latest_eval_human_normalized_score_mean"] == pytest.approx(
        33.0
    )
    assert aggregates_by_preset["ppo_breakout"]["delta_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(12.0)
    assert aggregates_by_preset["ppo_breakout"][
        "delta_vs_baseline_latest_eval_human_normalized_score_mean"
    ] == pytest.approx(12.0)
    assert aggregates_by_preset["ppo_breakout"]["ratio_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(
        72.0 / 60.0
    )
    assert aggregates_by_preset["ppo_breakout"]["ratio_vs_baseline_latest_eval_human_normalized_score_mean"] == (
        pytest.approx(45.0 / 33.0)
    )


def test_zoo_cli_report_json_includes_baseline_summary_top_movers_and_regressions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return, best_hns in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0, 46.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0, 44.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 70.0, 40.0, 72.0, 45.0),
        ("ppo__ALE-Breakout-v5__seed5__demo", "ppo", "ppo_breakout", 5, 74.0, 50.0, 74.0, 46.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
            "--baseline-preset",
            "dqn_breakout",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = payload["baseline_summary"]

    assert exit_code == 0
    assert summary["baseline_preset"] == "dqn_breakout"
    assert [entry["preset_name"] for entry in summary["top_movers_by_return_delta"]] == [
        "ppo_breakout",
        "ppg_breakout",
        "a2c_breakout",
    ]
    assert [entry["preset_name"] for entry in summary["top_regressions_by_return_delta"]] == [
        "a2c_breakout",
        "ppg_breakout",
        "ppo_breakout",
    ]
    assert [entry["preset_name"] for entry in summary["top_movers_by_normalized_delta"]] == [
        "ppo_breakout",
        "ppg_breakout",
        "a2c_breakout",
    ]
    assert [entry["preset_name"] for entry in summary["top_regressions_by_normalized_delta"]] == [
        "a2c_breakout",
        "ppg_breakout",
        "ppo_breakout",
    ]
    assert summary["top_movers_by_return_delta"][0]["delta_vs_baseline_latest_eval_return_mean_mean"] == pytest.approx(
        12.0
    )
    assert summary["top_regressions_by_normalized_delta"][0][
        "delta_vs_baseline_latest_eval_human_normalized_score_mean"
    ] == pytest.approx(-13.0)


def test_zoo_cli_report_json_min_seeds_filters_aggregates_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / "runs"
    for run_id, algo, preset_name, seed, latest_return, latest_hns, best_return in [
        ("dqn__ALE-Breakout-v5__seed7__demo", "dqn", "dqn_breakout", 7, 55.0, 30.0, 80.0),
        ("dqn__ALE-Breakout-v5__seed9__demo", "dqn", "dqn_breakout", 9, 65.0, 36.0, 78.0),
        ("ppo__ALE-Breakout-v5__seed3__demo", "ppo", "ppo_breakout", 3, 58.0, 40.0, 60.0),
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

    exit_code = zoo_main(
        [
            "--manifest",
            str(REPO_ROOT / "zoo" / "atari" / "benchmark.yaml"),
            "--format",
            "report",
            "--runs-dir",
            str(runs_dir),
            "--report-output",
            "json",
            "--group-by",
            "preset",
            "--min-seeds",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["min_seeds"] == 2
    assert len(payload["runs"]) == 3
    assert {run["preset_name"] for run in payload["runs"]} == {"dqn_breakout", "ppo_breakout"}
    assert len(payload["aggregates"]) == 1
    assert payload["aggregates"][0]["preset_name"] == "dqn_breakout"
    assert payload["aggregates"][0]["seed_count"] == 2
