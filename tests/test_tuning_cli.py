import csv
import io
import json
from pathlib import Path

import pytest
import yaml

from axiomrl.cli import main


def _write_base_train_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ppo-cartpole.yaml"
    config_path.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 17",
                "total_timesteps: 32",
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
    return config_path


def _write_study_report_fixture(tmp_path: Path) -> Path:
    study_root = tmp_path / "studies"
    study_dir = study_root / "fixture_report"
    study_dir.mkdir(parents=True)
    trial_a_dir = study_dir / "trials" / "trial-a"
    trial_b_dir = study_dir / "trials" / "trial-b"
    trial_c_dir = study_dir / "trials" / "trial-c"
    trial_a_dir.mkdir(parents=True)
    trial_b_dir.mkdir(parents=True)
    trial_c_dir.mkdir(parents=True)
    (trial_a_dir / "config.yaml").write_text(
        json.dumps({"total_timesteps": 32}, indent=2),
        encoding="utf-8",
    )
    (trial_c_dir / "config.yaml").write_text(
        json.dumps({"total_timesteps": 128}, indent=2),
        encoding="utf-8",
    )
    records = [
        {
            "trial_index": 0,
            "status": "completed",
            "params": {"total_timesteps": 32},
            "objective_value": 10.0,
            "run_dir": str(trial_a_dir),
            "checkpoint_path": str(trial_a_dir / "best.pt"),
            "error": None,
            "started_at": "2026-04-02T00:00:00+00:00",
            "ended_at": "2026-04-02T00:01:00+00:00",
        },
        {
            "trial_index": 1,
            "status": "failed",
            "params": {"total_timesteps": 64},
            "objective_value": None,
            "run_dir": None,
            "checkpoint_path": None,
            "error": "RuntimeError: boom",
            "started_at": "2026-04-02T00:02:00+00:00",
            "ended_at": "2026-04-02T00:03:00+00:00",
        },
        {
            "trial_index": 2,
            "status": "completed",
            "params": {"total_timesteps": 128},
            "objective_value": 30.0,
            "run_dir": str(trial_c_dir),
            "checkpoint_path": str(trial_c_dir / "best.pt"),
            "error": None,
            "started_at": "2026-04-02T00:04:00+00:00",
            "ended_at": "2026-04-02T00:05:00+00:00",
        },
    ]
    (study_dir / "study.json").write_text(
        json.dumps(
            {
                "study_name": "fixture_report",
                "backend": "native",
                "sampler": "grid",
                "objective": {"metric": "eval_return_mean", "mode": "max"},
                "base_config_path": str(tmp_path / "base.yaml"),
                "output_dir": str(study_root),
                "trial_count": 3,
                "status_counts": {"completed": 2, "failed": 1},
                "best_trial_index": 2,
                "best_objective_value": 30.0,
                "best_run_dir": str(trial_c_dir),
                "best_checkpoint_path": str(trial_c_dir / "best.pt"),
                "study_config": {
                    "base_config": str(tmp_path / "base.yaml"),
                    "output_dir": str(study_root),
                    "study": {
                        "name": "fixture_report",
                        "backend": "native",
                        "sampler": "grid",
                        "num_trials": None,
                        "seed": 0,
                        "fail_fast": False,
                        "objective": {"metric": "eval_return_mean", "mode": "max"},
                    },
                    "search_space": {
                        "total_timesteps": {"type": "int", "low": 32, "high": 128, "step": 32},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return study_dir


def test_tune_command_runs_study_and_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_root = tmp_path / "studies"
    study_config_path = tmp_path / "study.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_cli_tune",
                "  backend: native",
                "  sampler: grid",
                "  objective:",
                "    metric: global_step",
                "    mode: max",
                "search_space:",
                "  total_timesteps:",
                "    type: int",
                "    low: 32",
                "    high: 64",
                "    step: 32",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["tune", "--config", str(study_config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"study_dir={study_root / 'ppo_cli_tune'}" in output
    assert "best_trial_index=1" in output
    assert "best_objective_value=64.0" in output


def test_tune_command_rejects_invalid_study_config_with_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-invalid.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {tmp_path / 'studies'}",
                "study:",
                "  name: invalid_tune",
                "  backend: native",
                "  sampler: random",
                "  num_trials: 2",
                "search_space:",
                "  total_timesteps:",
                "    type: categorical",
                "    values: [32, 64]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(["tune", "--config", str(study_config_path)])

    assert exc.value.code == 2
    assert "objective" in capsys.readouterr().err


def test_tune_command_can_resume_existing_study(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_root = tmp_path / "studies"
    study_config_path = tmp_path / "study-resume.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_resume_tune",
                "  backend: native",
                "  sampler: grid",
                "  objective:",
                "    metric: global_step",
                "    mode: max",
                "search_space:",
                "  total_timesteps:",
                "    type: int",
                "    low: 32",
                "    high: 64",
                "    step: 32",
            ]
        ),
        encoding="utf-8",
    )

    first_exit_code = main(["tune", "--config", str(study_config_path)])
    assert first_exit_code == 0
    capsys.readouterr()

    study_dir = study_root / "ppo_resume_tune"
    resume_exit_code = main(["tune", "--resume-study", str(study_dir)])

    assert resume_exit_code == 0
    output = capsys.readouterr().out
    assert f"study_dir={study_dir}" in output
    assert "best_trial_index=1" in output


def test_tune_command_reports_missing_optuna_dependency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_root = tmp_path / "studies"
    study_config_path = tmp_path / "study-optuna.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_optuna_tune",
                "  backend: optuna",
                "  sampler: random",
                "  num_trials: 1",
                "  objective:",
                "    metric: global_step",
                "    mode: max",
                "search_space:",
                "  total_timesteps:",
                "    type: categorical",
                "    values: [32, 64]",
            ]
        ),
        encoding="utf-8",
    )

    import axiomrl.tuning.optuna_backend as optuna_backend

    def _raise_missing_dependency() -> object:
        raise ModuleNotFoundError('Optuna backend requires `pip install "axiomrl[tuning]"`')

    monkeypatch.setattr(optuna_backend, "_import_optuna", _raise_missing_dependency)

    with pytest.raises(SystemExit) as exc:
        main(["tune", "--config", str(study_config_path)])

    assert exc.value.code == 2
    assert "axiomrl[tuning]" in capsys.readouterr().err


def test_tune_report_command_can_emit_json_and_write_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_root = tmp_path / "studies"
    study_config_path = tmp_path / "study-report-cli.yaml"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_cli_report",
                "  backend: native",
                "  sampler: grid",
                "  objective:",
                "    metric: global_step",
                "    mode: max",
                "search_space:",
                "  total_timesteps:",
                "    type: int",
                "    low: 32",
                "    high: 64",
                "    step: 32",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["tune", "--config", str(study_config_path)]) == 0
    capsys.readouterr()

    study_dir = study_root / "ppo_cli_report"
    report_path = tmp_path / "reports" / "study-report.json"
    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--report-output",
            "json",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    written_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["study_name"] == "ppo_cli_report"
    assert payload["trial_count"] == 2
    assert payload["trials"][1]["trial_index"] == 1
    assert payload["selected_best_trial_index"] == 1
    assert payload["selected_best_objective_value"] == 64.0
    assert payload["selected_status_counts"] == {"completed": 2}
    assert payload["selected_objective_summary"] == {
        "completed_trials": 2,
        "failed_trials": 0,
        "min": 32.0,
        "max": 64.0,
        "mean": 48.0,
        "median": 48.0,
    }
    assert payload["search_efficiency_summary"] == {
        "selected_trial_count": 2,
        "completed_trials": 2,
        "failed_trials": 0,
        "failure_rate": 0.0,
        "selected_best_trial_index": 1,
        "selected_best_objective_value": 64.0,
        "selected_trials_until_best": 2,
        "selected_trial_share_until_best": 1.0,
        "completed_trials_until_best": 2,
        "completed_trial_share_until_best": 1.0,
        "time_to_best_seconds": payload["search_efficiency_summary"]["time_to_best_seconds"],
        "best_vs_median_delta": 16.0,
        "best_vs_mean_delta": 16.0,
        "lowest_coverage_parameter": {
            "name": "total_timesteps",
            "coverage_ratio": 1.0,
            "candidate_count": 2,
            "observed_unique_count": 2,
        },
        "highest_coverage_parameter": {
            "name": "total_timesteps",
            "coverage_ratio": 1.0,
            "candidate_count": 2,
            "observed_unique_count": 2,
        },
    }
    assert (
        payload["search_efficiency_summary"]["time_to_best_seconds"] is None
        or payload["search_efficiency_summary"]["time_to_best_seconds"] >= 0.0
    )
    assert payload["selected_incumbent_trace"] == [
        {
            "trial_index": 0,
            "status": "completed",
            "objective_value": 32.0,
            "selected_incumbent_trial_index": 0,
            "selected_incumbent_objective_value": 32.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": None,
            "selected_incumbent_trials_since_previous_update": None,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
        {
            "trial_index": 1,
            "status": "completed",
            "objective_value": 64.0,
            "selected_incumbent_trial_index": 1,
            "selected_incumbent_objective_value": 64.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": 32.0,
            "selected_incumbent_trials_since_previous_update": 1,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
    ]
    assert payload["selected_incumbent_update_summary"] == {
        "incumbent_update_count": 2,
        "first_incumbent_trial_index": 0,
        "latest_incumbent_trial_index": 1,
        "latest_incumbent_objective_value": 64.0,
        "mean_improvement_over_previous": 32.0,
        "max_improvement_over_previous": 32.0,
        "mean_trials_since_previous_update": 1.0,
        "max_trials_since_previous_update": 1,
    }
    assert payload["selected_incumbent_staleness_summary"] == {
        "latest_incumbent_age_trials": 0,
        "latest_incumbent_age_seconds": 0.0,
        "max_incumbent_age_trials": 0,
        "max_incumbent_age_seconds": 0.0,
    }
    assert any(
        entry["trial_index"] == 1
        and entry["objective_value"] == 64.0
        and entry["selected_best_objective_delta"] == 0.0
        and entry["params"] == {"total_timesteps": 64}
        and (entry["duration_seconds"] is None or entry["duration_seconds"] >= 0.0)
        for entry in payload["selected_objective_duration_frontier"]
    )
    assert payload["selected_parameter_summaries"]["total_timesteps"]["selected_best_value"] == 64
    assert payload["selected_parameter_summaries"]["total_timesteps"]["candidate_count"] == 2
    assert payload["selected_parameter_summaries"]["total_timesteps"]["coverage_ratio"] == 1.0
    assert payload["selected_parameter_incumbent_summaries"] == {
        "total_timesteps": {
            "incumbent_update_count": 2,
            "contributing_values": [32, 64],
            "contributing_value_count": 2,
            "top_incumbent_value": 64,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 64,
            "latest_incumbent_trial_index": 1,
        }
    }
    assert payload["selected_parameter_incumbent_leaderboard"] == [
        {
            "name": "total_timesteps",
            "incumbent_update_count": 2,
            "contributing_values": [32, 64],
            "contributing_value_count": 2,
            "top_incumbent_value": 64,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 64,
            "latest_incumbent_trial_index": 1,
        }
    ]
    assert payload["selected_parameter_effect_leaderboard"] == [
        {
            "name": "total_timesteps",
            "observed_value_count": 2,
            "completed_value_count": 2,
            "best_objective_spread": 32.0,
            "mean_objective_spread": 32.0,
            "top_value_by_best_objective": 64,
            "top_best_objective_value": 64.0,
            "bottom_value_by_best_objective": 32,
            "bottom_best_objective_value": 32.0,
            "top_value_by_mean_objective": 64,
            "top_mean_objective_value": 64.0,
            "bottom_value_by_mean_objective": 32,
            "bottom_mean_objective_value": 32.0,
        }
    ]
    assert [
        {
            "value": entry["value"],
            "trial_count": entry["trial_count"],
            "completed_trials": entry["completed_trials"],
            "failed_trials": entry["failed_trials"],
            "completion_rate": entry["completion_rate"],
            "failure_rate": entry["failure_rate"],
            "best_objective_value": entry["best_objective_value"],
            "mean_objective_value": entry["mean_objective_value"],
            "median_objective_value": entry["median_objective_value"],
            "incumbent_updates": entry["incumbent_updates"],
            "latest_incumbent_trial_index": entry["latest_incumbent_trial_index"],
            "selected_best_objective_delta": entry["selected_best_objective_delta"],
            "rank_by_best_objective_value": entry["rank_by_best_objective_value"],
            "rank_by_mean_objective_value": entry["rank_by_mean_objective_value"],
        }
        for entry in payload["selected_parameter_value_summaries"]["total_timesteps"]
    ] == [
        {
            "value": 32,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 32.0,
            "mean_objective_value": 32.0,
            "median_objective_value": 32.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 0,
            "selected_best_objective_delta": 32.0,
            "rank_by_best_objective_value": 2,
            "rank_by_mean_objective_value": 2,
        },
        {
            "value": 64,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 64.0,
            "mean_objective_value": 64.0,
            "median_objective_value": 64.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 1,
            "selected_best_objective_delta": 0.0,
            "rank_by_best_objective_value": 1,
            "rank_by_mean_objective_value": 1,
        },
    ]
    for entry in payload["selected_parameter_value_summaries"]["total_timesteps"]:
        assert entry["timed_trials"] == 1
        assert entry["untimed_trials"] == 0
        assert entry["mean_duration_seconds"] >= 0.0
        assert entry["min_duration_seconds"] == pytest.approx(entry["mean_duration_seconds"])
        assert entry["max_duration_seconds"] == pytest.approx(entry["mean_duration_seconds"])
        assert entry["median_duration_seconds"] == pytest.approx(entry["mean_duration_seconds"])
    assert [trial["selected_best_objective_delta"] for trial in payload["trials"]] == [32.0, 0.0]
    assert [trial["selected_incumbent_trial_index"] for trial in payload["trials"]] == [0, 1]
    assert [trial["selected_incumbent_objective_value"] for trial in payload["trials"]] == [32.0, 64.0]
    assert [trial["selected_is_incumbent_update"] for trial in payload["trials"]] == [True, True]
    assert [trial["selected_incumbent_update_improvement"] for trial in payload["trials"]] == [None, 32.0]
    assert [trial["selected_incumbent_trials_since_previous_update"] for trial in payload["trials"]] == [None, 1]
    assert [trial["selected_incumbent_age_trials"] for trial in payload["trials"]] == [0, 0]
    assert [trial["selected_incumbent_age_seconds"] for trial in payload["trials"]] == [0.0, 0.0]
    assert payload["trials"][1]["is_objective_duration_frontier"] is True
    assert written_payload["best_trial_index"] == 1


def test_tune_report_command_supports_status_sort_and_top_k_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--status",
            "completed",
            "--sort-by",
            "objective-value",
            "--descending",
            "--top-k",
            "1",
            "--report-output",
            "csv",
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 1
    assert rows[0]["trial_index"] == "2"
    assert rows[0]["status"] == "completed"
    assert rows[0]["objective_value"] == "30.0"
    assert rows[0]["selected_best_trial_index"] == "2"
    assert rows[0]["selected_best_objective_delta"] == "0.0"


def test_tune_report_command_supports_duration_sorting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    trials = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trials[0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    trials[1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    trials[2]["ended_at"] = "2026-04-02T00:05:30+00:00"
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in trials) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--sort-by",
            "duration-seconds",
            "--descending",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [trial["trial_index"] for trial in payload["trials"]] == [1, 2, 0]
    assert [trial["duration_seconds"] for trial in payload["trials"]] == [90.0, 90.0, 30.0]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "duration-seconds",
        "descending": True,
        "top_k": None,
    }


def test_tune_report_command_supports_parameter_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--status",
            "completed",
            "--param",
            "total_timesteps=128",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_best_trial_index"] == 2
    assert payload["report_filters"] == {
        "status": "completed",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "param_filters": {
            "total_timesteps": 128,
        },
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [2]


def test_tune_report_command_supports_objective_ceiling_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--objective-at-most",
            "15",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_best_trial_index"] == 0
    assert payload["selected_best_objective_value"] == 10.0
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "objective_at_most": 15.0,
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [0]


def test_tune_report_command_supports_duration_floor_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    trials = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trials[0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    trials[1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    trials[2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in trials) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--status",
            "completed",
            "--duration-at-least",
            "45",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_best_trial_index"] == 2
    assert payload["selected_best_objective_value"] == 30.0
    assert payload["report_filters"] == {
        "status": "completed",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "duration_at_least": 45.0,
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [2]


def test_tune_report_command_supports_frontier_only_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    trials = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trials[0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    trials[1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    trials[2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in trials) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--frontier-only",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 2
    assert payload["selected_status_counts"] == {"completed": 2}
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "frontier_only": True,
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [0, 2]
    assert [trial["is_objective_duration_frontier"] for trial in payload["trials"]] == [True, True]
    assert [entry["trial_index"] for entry in payload["selected_objective_duration_frontier"]] == [0, 2]


def test_tune_report_command_supports_error_substring_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--error-contains",
            "runtimeerror",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_status_counts"] == {"failed": 1}
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error_contains": "runtimeerror",
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [1]


def test_tune_report_command_supports_exact_error_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--error",
            "RuntimeError: boom",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_status_counts"] == {"failed": 1}
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error": "RuntimeError: boom",
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [1]


def test_tune_report_command_supports_error_type_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--error-type",
            "runtimeerror",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_trial_count"] == 1
    assert payload["selected_status_counts"] == {"failed": 1}
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error_type": "runtimeerror",
    }
    assert [trial["trial_index"] for trial in payload["trials"]] == [1]


def test_tune_report_command_rejects_combined_error_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "tune-report",
                "--study-dir",
                str(study_dir),
                "--error",
                "RuntimeError: boom",
                "--error-contains",
                "runtimeerror",
            ]
        )

    assert exc.value.code == 2
    assert "--error and --error-contains cannot be used together" in capsys.readouterr().err


def test_tune_report_command_rejects_inverted_objective_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "tune-report",
                "--study-dir",
                str(study_dir),
                "--objective-at-least",
                "30",
                "--objective-at-most",
                "10",
            ]
        )

    assert exc.value.code == 2
    assert "--objective-at-least cannot be greater than --objective-at-most" in capsys.readouterr().err


def test_tune_report_command_rejects_inverted_duration_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "tune-report",
                "--study-dir",
                str(study_dir),
                "--duration-at-least",
                "30",
                "--duration-at-most",
                "10",
            ]
        )

    assert exc.value.code == 2
    assert "--duration-at-least cannot be greater than --duration-at-most" in capsys.readouterr().err


def test_tune_report_command_includes_selected_error_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_error_summaries"] == [
        {
            "error": "RuntimeError: boom",
            "failed_trials": 1,
            "selected_trial_share": 0.3333333333,
            "failed_trial_share": 1.0,
            "trial_indices": [1],
        }
    ]
    assert payload["selected_error_type_summaries"] == [
        {
            "error_type": "RuntimeError",
            "errors": ["RuntimeError: boom"],
            "failed_trials": 1,
            "selected_trial_share": 0.3333333333,
            "failed_trial_share": 1.0,
            "trial_indices": [1],
        }
    ]
    assert payload["selected_duration_summary"] == {
        "timed_trials": 3,
        "untimed_trials": 0,
        "min_seconds": 60.0,
        "max_seconds": 60.0,
        "mean_seconds": 60.0,
        "median_seconds": 60.0,
    }


def test_tune_report_command_supports_focus_param(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--focus-param",
            "total_timesteps",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in payload["focused_parameter_value_summary"]] == [128, 32, 64]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "best-objective-value",
    }


def test_tune_report_command_supports_focus_param_sorting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--focus-param",
            "total_timesteps",
            "--focus-sort-by",
            "value",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in payload["focused_parameter_value_summary"]] == [32, 64, 128]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "value",
    }


def test_tune_report_command_supports_focus_param_incumbent_update_sorting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--focus-param",
            "total_timesteps",
            "--focus-sort-by",
            "incumbent-updates",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in payload["focused_parameter_value_summary"]] == [128, 32, 64]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "incumbent-updates",
    }


def test_tune_report_command_supports_focus_param_duration_sorting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    trials = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trials[0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    trials[1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    trials[2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in trials) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--focus-param",
            "total_timesteps",
            "--focus-sort-by",
            "mean-duration-seconds",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in payload["focused_parameter_value_summary"]] == [32, 128, 64]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "mean-duration-seconds",
    }


def test_tune_report_command_supports_focus_param_top_k(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--focus-param",
            "total_timesteps",
            "--focus-top-k",
            "2",
            "--report-output",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in payload["focused_parameter_value_summary"]] == [128, 32]
    assert payload["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "best-objective-value",
        "focus_top_k": 2,
    }


def test_tune_report_command_rejects_focus_only_flags_without_focus_param(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "tune-report",
                "--study-dir",
                str(study_dir),
                "--focus-sort-by",
                "value",
            ]
        )

    assert exc.value.code == 2
    assert "--focus-sort-by requires --focus-param" in capsys.readouterr().err


def test_tune_report_command_can_export_selected_configs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    export_dir = tmp_path / "exported-configs"

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--status",
            "completed",
            "--sort-by",
            "objective-value",
            "--descending",
            "--top-k",
            "1",
            "--report-output",
            "json",
            "--export-configs-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    exported_config_path = export_dir / "rank-001_trial-0002.yaml"
    manifest_path = export_dir / "manifest.json"

    assert payload["config_export_summary"] == {
        "output_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "exported_count": 1,
        "skipped_trial_indices": [],
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["exported_trials"][0]["trial_index"] == 2
    assert yaml.safe_load(exported_config_path.read_text(encoding="utf-8"))["total_timesteps"] == 128


def test_tune_report_command_exports_only_completed_trials(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    failed_run_dir = study_dir / "trials" / "trial-b"
    (failed_run_dir / "config.yaml").write_text(
        json.dumps({"total_timesteps": 64}, indent=2),
        encoding="utf-8",
    )
    trials = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trials[1]["run_dir"] = str(failed_run_dir)
    trials[1]["checkpoint_path"] = str(failed_run_dir / "best.pt")
    (study_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in trials) + "\n",
        encoding="utf-8",
    )
    export_dir = tmp_path / "exported-configs"

    exit_code = main(
        [
            "tune-report",
            "--study-dir",
            str(study_dir),
            "--report-output",
            "json",
            "--export-configs-dir",
            str(export_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_export_summary"] == {
        "output_dir": str(export_dir),
        "manifest_path": str(export_dir / "manifest.json"),
        "exported_count": 2,
        "skipped_trial_indices": [],
    }
    assert (export_dir / "rank-001_trial-0000.yaml").exists()
    assert not (export_dir / "rank-002_trial-0001.yaml").exists()
    assert (export_dir / "rank-002_trial-0002.yaml").exists()
    manifest_payload = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [trial["trial_index"] for trial in manifest_payload["exported_trials"]] == [0, 2]
