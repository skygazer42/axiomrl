import json
from pathlib import Path

import pytest
import yaml

from rl_training.tuning.config import load_study_config, serialize_study_config
from rl_training.tuning.study import (
    csv_study_report_rows,
    export_selected_study_configs,
    load_study_report,
    render_csv_study_report,
    render_text_study_report,
    resume_study,
    run_study,
    select_study_report,
)


def _write_base_train_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "ppo-cartpole.yaml"
    config_path.write_text(
        "\n".join(
            [
                "algo: ppo",
                "env_id: CartPole-v1",
                "seed: 11",
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
    run_a = study_dir / "trials" / "trial-a"
    run_b = study_dir / "trials" / "trial-b"
    run_c = study_dir / "trials" / "trial-c"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    run_c.mkdir(parents=True)
    (run_a / "config.yaml").write_text(
        json.dumps(
            {
                "total_timesteps": 32,
                "algo_kwargs": {"learning_rate": 0.001},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_c / "config.yaml").write_text(
        json.dumps(
            {
                "total_timesteps": 128,
                "algo_kwargs": {"learning_rate": 0.0003},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    records = [
        {
            "trial_index": 0,
            "status": "completed",
            "params": {"total_timesteps": 32, "algo_kwargs.learning_rate": 0.001},
            "objective_value": 10.0,
            "run_dir": str(run_a),
            "checkpoint_path": str(run_a / "best.pt"),
            "error": None,
            "started_at": "2026-04-02T00:00:00+00:00",
            "ended_at": "2026-04-02T00:01:00+00:00",
        },
        {
            "trial_index": 1,
            "status": "failed",
            "params": {"total_timesteps": 64, "algo_kwargs.learning_rate": 0.01},
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
            "params": {"total_timesteps": 128, "algo_kwargs.learning_rate": 0.0003},
            "objective_value": 30.0,
            "run_dir": str(run_c),
            "checkpoint_path": str(run_c / "best.pt"),
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
                "best_run_dir": str(run_c),
                "best_checkpoint_path": str(run_c / "best.pt"),
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
                        "algo_kwargs.learning_rate": {
                            "type": "categorical",
                            "values": [0.0003, 0.001, 0.01],
                        },
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


def test_run_study_writes_native_grid_artifacts_and_best_config(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_grid_tune",
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

    result = run_study(load_study_config(study_config_path))

    study_dir = study_root / "ppo_grid_tune"
    trials_dir = study_dir / "trials"

    assert result.study_dir == study_dir
    assert result.best_trial_index == 1
    assert result.best_objective_value == 64.0
    assert (study_dir / "study.json").exists()
    assert (study_dir / "trials.jsonl").exists()
    assert (study_dir / "best_trial.json").exists()
    assert (study_dir / "best_config.yaml").exists()
    assert trials_dir.exists()
    assert len([path for path in trials_dir.iterdir() if path.is_dir()]) == 2

    study_payload = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    assert study_payload["best_trial_index"] == 1
    assert study_payload["status_counts"]["completed"] == 2

    best_config_payload = yaml.safe_load((study_dir / "best_config.yaml").read_text(encoding="utf-8"))
    assert best_config_payload["total_timesteps"] == 64


def test_run_study_writes_native_random_trials_for_requested_count(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-random.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_random_tune",
                "  backend: native",
                "  sampler: random",
                "  num_trials: 3",
                "  seed: 5",
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

    result = run_study(load_study_config(study_config_path))

    study_dir = study_root / "ppo_random_tune"
    study_payload = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    trial_lines = (study_dir / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert result.study_dir == study_dir
    assert len(trial_lines) == 3
    assert study_payload["trial_count"] == 3
    assert study_payload["status_counts"]["completed"] == 3


def test_load_study_report_includes_trial_records_and_status_counts(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-report.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_report_tune",
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

    run_study(load_study_config(study_config_path))

    report = load_study_report(study_root / "ppo_report_tune")

    assert report["study_name"] == "ppo_report_tune"
    assert report["study_dir"] == str(study_root / "ppo_report_tune")
    assert report["trial_count"] == 2
    assert report["status_counts"] == {"completed": 2}
    assert report["best_trial_index"] == 1
    assert len(report["trials"]) == 2
    assert report["trials"][0]["trial_index"] == 0
    assert report["trials"][1]["objective_value"] == 64.0


def test_csv_study_report_rows_and_renderer_include_trial_parameters(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-report-csv.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_report_csv",
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

    run_study(load_study_config(study_config_path))

    payload = select_study_report(load_study_report(study_root / "ppo_report_csv"))
    rows = csv_study_report_rows(payload)
    rendered = render_csv_study_report(payload)

    assert len(rows) == 2
    assert rows[0]["trial_index"] == 0
    assert rows[1]["trial_index"] == 1
    assert rows[1]["param_total_timesteps"] == 64
    assert "total_timesteps" in rows[0]["selected_parameter_value_summaries_json"]
    assert "study_name,backend,sampler,objective_metric,objective_mode" in rendered
    assert "selected_parameter_value_summaries_json" in rendered
    assert "param_total_timesteps" in rendered
    assert "64" in rendered


def test_focus_param_text_and_csv_renderers_surface_focused_bucket_details(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir), focus_param="total_timesteps")
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert rows[0]["focused_parameter_name"] == "total_timesteps"
    assert rows[0]["focused_parameter_value"] == 32
    assert rows[0]["focused_parameter_rank_by_best_objective_value"] == 2
    assert rows[0]["focused_parameter_mean_duration_seconds"] == 60.0
    assert rows[0]["focused_parameter_incumbent_updates"] == 1
    assert rows[0]["focused_parameter_latest_incumbent_trial_index"] == 0
    assert rows[2]["focused_parameter_value"] == 128
    assert rows[2]["focused_parameter_rank_by_best_objective_value"] == 1
    assert rows[2]["focused_parameter_incumbent_updates"] == 1
    assert rows[2]["focused_parameter_latest_incumbent_trial_index"] == 2
    assert "focused_parameter_rank_by_best_objective_value" in rendered_csv
    assert "focused_parameter_mean_duration_seconds" in rendered_csv
    assert "focused_parameter_incumbent_updates" in rendered_csv
    assert "focused_parameter_latest_incumbent_trial_index" in rendered_csv
    assert "focused_parameter_value" in rendered_csv
    assert "[focused parameter total_timesteps]" in rendered_text
    assert "rank_by_best_objective_value=1" in rendered_text
    assert "mean_duration_seconds=60.0" in rendered_text
    assert "incumbent_updates=1" in rendered_text
    assert "latest_incumbent_trial_index=2" in rendered_text
    assert "value=128" in rendered_text


def test_study_report_renderers_surface_selected_error_summaries(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert "RuntimeError: boom" in str(rows[0]["selected_error_summaries_json"])
    assert "RuntimeError" in str(rows[0]["selected_error_type_summaries_json"])
    assert '"mean_seconds": 60.0' in str(rows[0]["selected_duration_summary_json"])
    assert "selected_error_summaries_json" in rendered_csv
    assert "selected_error_type_summaries_json" in rendered_csv
    assert "selected_duration_summary_json" in rendered_csv
    assert "duration_seconds" in rendered_csv
    assert "selected_error_summaries=" in rendered_text
    assert "selected_error_type_summaries=" in rendered_text
    assert "selected_duration_summary=" in rendered_text
    assert "[failed trial errors]" in rendered_text
    assert "[failed trial error types]" in rendered_text
    assert "error=RuntimeError: boom" in rendered_text
    assert "error_type=RuntimeError" in rendered_text
    assert "duration_seconds=60.0" in rendered_text


def test_study_report_renderers_flatten_search_efficiency_convergence_fields(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert rows[0]["search_efficiency_selected_trials_until_best"] == 3
    assert rows[0]["search_efficiency_selected_trial_share_until_best"] == 1.0
    assert rows[0]["search_efficiency_completed_trials_until_best"] == 2
    assert rows[0]["search_efficiency_completed_trial_share_until_best"] == 1.0
    assert rows[0]["search_efficiency_time_to_best_seconds"] == 300.0
    assert "search_efficiency_selected_trials_until_best" in rendered_csv
    assert "search_efficiency_time_to_best_seconds" in rendered_csv
    assert "search_efficiency_selected_trials_until_best=3" in rendered_text
    assert "search_efficiency_completed_trials_until_best=2" in rendered_text
    assert "search_efficiency_time_to_best_seconds=300.0" in rendered_text


def test_study_report_renderers_surface_selected_incumbent_trace(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert json.loads(str(rows[0]["selected_incumbent_trace_json"])) == [
        {
            "trial_index": 0,
            "status": "completed",
            "objective_value": 10.0,
            "selected_incumbent_trial_index": 0,
            "selected_incumbent_objective_value": 10.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": None,
            "selected_incumbent_trials_since_previous_update": None,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
        {
            "trial_index": 1,
            "status": "failed",
            "objective_value": None,
            "selected_incumbent_trial_index": 0,
            "selected_incumbent_objective_value": 10.0,
            "selected_is_incumbent_update": False,
            "selected_incumbent_update_improvement": None,
            "selected_incumbent_trials_since_previous_update": None,
            "selected_incumbent_age_trials": 1,
            "selected_incumbent_age_seconds": 120.0,
        },
        {
            "trial_index": 2,
            "status": "completed",
            "objective_value": 30.0,
            "selected_incumbent_trial_index": 2,
            "selected_incumbent_objective_value": 30.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": 20.0,
            "selected_incumbent_trials_since_previous_update": 2,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
    ]
    assert json.loads(str(rows[0]["selected_incumbent_update_summary_json"])) == {
        "incumbent_update_count": 2,
        "first_incumbent_trial_index": 0,
        "latest_incumbent_trial_index": 2,
        "latest_incumbent_objective_value": 30.0,
        "mean_improvement_over_previous": 20.0,
        "max_improvement_over_previous": 20.0,
        "mean_trials_since_previous_update": 2.0,
        "max_trials_since_previous_update": 2,
    }
    assert json.loads(str(rows[0]["selected_incumbent_staleness_summary_json"])) == {
        "latest_incumbent_age_trials": 0,
        "latest_incumbent_age_seconds": 0.0,
        "max_incumbent_age_trials": 1,
        "max_incumbent_age_seconds": 120.0,
    }
    assert rows[1]["selected_incumbent_trial_index"] == 0
    assert rows[1]["selected_incumbent_objective_value"] == 10.0
    assert rows[1]["selected_is_incumbent_update"] is False
    assert rows[1]["selected_incumbent_update_improvement"] is None
    assert rows[1]["selected_incumbent_trials_since_previous_update"] is None
    assert rows[1]["selected_incumbent_age_trials"] == 1
    assert rows[1]["selected_incumbent_age_seconds"] == 120.0
    assert rows[2]["selected_incumbent_trial_index"] == 2
    assert rows[2]["selected_incumbent_objective_value"] == 30.0
    assert rows[2]["selected_is_incumbent_update"] is True
    assert rows[2]["selected_incumbent_update_improvement"] == 20.0
    assert rows[2]["selected_incumbent_trials_since_previous_update"] == 2
    assert rows[2]["selected_incumbent_age_trials"] == 0
    assert rows[2]["selected_incumbent_age_seconds"] == 0.0
    assert "selected_incumbent_update_summary_json" in rendered_csv
    assert "selected_incumbent_staleness_summary_json" in rendered_csv
    assert "selected_incumbent_trace_json" in rendered_csv
    assert "selected_incumbent_trial_index" in rendered_csv
    assert "selected_is_incumbent_update" in rendered_csv
    assert "selected_incumbent_update_improvement" in rendered_csv
    assert "selected_incumbent_trials_since_previous_update" in rendered_csv
    assert "selected_incumbent_age_trials" in rendered_csv
    assert "selected_incumbent_age_seconds" in rendered_csv
    assert "selected_incumbent_update_summary=" in rendered_text
    assert "selected_incumbent_staleness_summary=" in rendered_text
    assert "selected_incumbent_trace=" in rendered_text
    assert "[incumbent trace]" in rendered_text
    assert "[incumbent step 2]" in rendered_text
    assert "selected_incumbent_objective_value=30.0" in rendered_text
    assert "selected_is_incumbent_update=True" in rendered_text
    assert "selected_incumbent_update_improvement=20.0" in rendered_text
    assert "selected_incumbent_trials_since_previous_update=2" in rendered_text
    assert "selected_incumbent_age_trials=1" in rendered_text
    assert "selected_incumbent_age_seconds=120.0" in rendered_text


def test_study_report_renderers_surface_objective_duration_frontier(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert json.loads(str(rows[0]["selected_objective_duration_frontier_json"])) == [
        {
            "trial_index": 2,
            "objective_value": 30.0,
            "duration_seconds": 60.0,
            "selected_best_objective_delta": 0.0,
            "params": {
                "algo_kwargs.learning_rate": 0.0003,
                "total_timesteps": 128,
            },
        }
    ]
    assert rows[0]["is_objective_duration_frontier"] is False
    assert rows[2]["is_objective_duration_frontier"] is True
    assert "selected_objective_duration_frontier_json" in rendered_csv
    assert "is_objective_duration_frontier" in rendered_csv
    assert "selected_objective_duration_frontier=" in rendered_text
    assert "[objective-duration frontier]" in rendered_text
    assert "[frontier trial 2]" in rendered_text
    assert "trial_index=2" in rendered_text
    assert "objective_value=30.0" in rendered_text
    assert "duration_seconds=60.0" in rendered_text


def test_study_report_renderers_surface_parameter_incumbent_summaries(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert json.loads(str(rows[0]["selected_parameter_incumbent_summaries_json"])) == {
        "algo_kwargs.learning_rate": {
            "contributing_value_count": 2,
            "contributing_values": [0.0003, 0.001],
            "incumbent_update_count": 2,
            "latest_incumbent_trial_index": 2,
            "latest_incumbent_value": 0.0003,
            "top_incumbent_value": 0.0003,
            "top_incumbent_value_updates": 1,
        },
        "total_timesteps": {
            "contributing_value_count": 2,
            "contributing_values": [32, 128],
            "incumbent_update_count": 2,
            "latest_incumbent_trial_index": 2,
            "latest_incumbent_value": 128,
            "top_incumbent_value": 128,
            "top_incumbent_value_updates": 1,
        },
    }
    assert "selected_parameter_incumbent_summaries_json" in rendered_csv
    assert "selected_parameter_incumbent_summary[total_timesteps]=" in rendered_text
    assert "selected_parameter_incumbent_summary[algo_kwargs.learning_rate]=" in rendered_text
    assert '"top_incumbent_value": 128' in rendered_text
    assert '"latest_incumbent_trial_index": 2' in rendered_text


def test_study_report_renderers_surface_parameter_incumbent_leaderboard(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert json.loads(str(rows[0]["selected_parameter_incumbent_leaderboard_json"])) == [
        {
            "name": "algo_kwargs.learning_rate",
            "incumbent_update_count": 2,
            "contributing_values": [0.0003, 0.001],
            "contributing_value_count": 2,
            "top_incumbent_value": 0.0003,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 0.0003,
            "latest_incumbent_trial_index": 2,
        },
        {
            "name": "total_timesteps",
            "incumbent_update_count": 2,
            "contributing_values": [32, 128],
            "contributing_value_count": 2,
            "top_incumbent_value": 128,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 128,
            "latest_incumbent_trial_index": 2,
        },
    ]
    assert "selected_parameter_incumbent_leaderboard_json" in rendered_csv
    assert "selected_parameter_incumbent_leaderboard=" in rendered_text
    assert "[parameter incumbent leaderboard]" in rendered_text
    assert "[parameter incumbent algo_kwargs.learning_rate]" in rendered_text
    assert "latest_incumbent_value=128" in rendered_text


def test_study_report_renderers_surface_parameter_effect_leaderboard(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(load_study_report(study_dir))
    rows = csv_study_report_rows(payload)
    rendered_csv = render_csv_study_report(payload)
    rendered_text = render_text_study_report(payload)

    assert json.loads(str(rows[0]["selected_parameter_effect_leaderboard_json"])) == [
        {
            "name": "algo_kwargs.learning_rate",
            "observed_value_count": 3,
            "completed_value_count": 2,
            "best_objective_spread": 20.0,
            "mean_objective_spread": 20.0,
            "top_value_by_best_objective": 0.0003,
            "top_best_objective_value": 30.0,
            "bottom_value_by_best_objective": 0.001,
            "bottom_best_objective_value": 10.0,
            "top_value_by_mean_objective": 0.0003,
            "top_mean_objective_value": 30.0,
            "bottom_value_by_mean_objective": 0.001,
            "bottom_mean_objective_value": 10.0,
        },
        {
            "name": "total_timesteps",
            "observed_value_count": 3,
            "completed_value_count": 2,
            "best_objective_spread": 20.0,
            "mean_objective_spread": 20.0,
            "top_value_by_best_objective": 128,
            "top_best_objective_value": 30.0,
            "bottom_value_by_best_objective": 32,
            "bottom_best_objective_value": 10.0,
            "top_value_by_mean_objective": 128,
            "top_mean_objective_value": 30.0,
            "bottom_value_by_mean_objective": 32,
            "bottom_mean_objective_value": 10.0,
        },
    ]
    assert "selected_parameter_effect_leaderboard_json" in rendered_csv
    assert "selected_parameter_effect_leaderboard=" in rendered_text
    assert "[parameter effect leaderboard]" in rendered_text
    assert "[parameter effect total_timesteps]" in rendered_text
    assert "best_objective_spread=20.0" in rendered_text
    assert "top_value_by_best_objective=128" in rendered_text


def test_select_study_report_can_filter_sort_and_limit_trials(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        status="completed",
        sort_by="objective-value",
        descending=True,
        top_k=1,
    )

    assert selected["trial_count"] == 3
    assert selected["selected_trial_count"] == 1
    assert selected["report_filters"] == {
        "status": "completed",
        "sort_by": "objective-value",
        "descending": True,
        "top_k": 1,
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [2]


def test_select_study_report_can_sort_by_duration_seconds(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["trials"][0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    payload["trials"][1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    payload["trials"][2]["ended_at"] = "2026-04-02T00:05:30+00:00"
    selected = select_study_report(
        payload,
        sort_by="duration-seconds",
        descending=True,
    )

    assert [trial["trial_index"] for trial in selected["trials"]] == [1, 2, 0]
    assert [trial["duration_seconds"] for trial in selected["trials"]] == [90.0, 90.0, 30.0]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "duration-seconds",
        "descending": True,
        "top_k": None,
    }


def test_select_study_report_can_filter_trials_by_parameter_values(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        status="completed",
        param_filters={
            "total_timesteps": 128,
            "algo_kwargs.learning_rate": 0.0003,
        },
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_best_trial_index"] == 2
    assert selected["selected_best_objective_value"] == 30.0
    assert selected["report_filters"] == {
        "status": "completed",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "param_filters": {
            "total_timesteps": 128,
            "algo_kwargs.learning_rate": 0.0003,
        },
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [2]


def test_select_study_report_can_filter_trials_by_objective_floor(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        objective_at_least=20.0,
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_best_trial_index"] == 2
    assert selected["selected_best_objective_value"] == 30.0
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "objective_at_least": 20.0,
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [2]


def test_select_study_report_can_filter_trials_by_duration_ceiling(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["trials"][0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    payload["trials"][1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    payload["trials"][2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    selected = select_study_report(
        payload,
        status="completed",
        duration_at_most=45.0,
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_best_trial_index"] == 0
    assert selected["selected_best_objective_value"] == 10.0
    assert selected["report_filters"] == {
        "status": "completed",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "duration_at_most": 45.0,
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [0]


def test_select_study_report_can_filter_to_objective_duration_frontier(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["trials"][0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    payload["trials"][1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    payload["trials"][2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    selected = select_study_report(
        payload,
        frontier_only=True,
    )

    assert selected["selected_trial_count"] == 2
    assert selected["selected_status_counts"] == {"completed": 2}
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "frontier_only": True,
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [0, 2]
    assert [trial["is_objective_duration_frontier"] for trial in selected["trials"]] == [True, True]
    assert [entry["trial_index"] for entry in selected["selected_objective_duration_frontier"]] == [0, 2]


def test_select_study_report_can_filter_trials_by_error_substring(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        error_contains="runtimeerror",
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_status_counts"] == {"failed": 1}
    assert selected["selected_error_summaries"] == [
        {
            "error": "RuntimeError: boom",
            "failed_trials": 1,
            "selected_trial_share": 1.0,
            "failed_trial_share": 1.0,
            "trial_indices": [1],
        }
    ]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error_contains": "runtimeerror",
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [1]


def test_select_study_report_can_filter_trials_by_exact_error(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        error="RuntimeError: boom",
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_status_counts"] == {"failed": 1}
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error": "RuntimeError: boom",
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [1]


def test_select_study_report_can_filter_trials_by_error_type(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        error_type="runtimeerror",
    )

    assert selected["selected_trial_count"] == 1
    assert selected["selected_status_counts"] == {"failed": 1}
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "error_type": "runtimeerror",
    }
    assert [trial["trial_index"] for trial in selected["trials"]] == [1]


def test_select_study_report_rejects_combined_error_filters(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)

    with pytest.raises(ValueError, match="--error and --error-contains cannot be used together"):
        select_study_report(
            payload,
            error="RuntimeError: boom",
            error_contains="runtimeerror",
        )


def test_select_study_report_rejects_inverted_objective_thresholds(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)

    with pytest.raises(ValueError, match="--objective-at-least cannot be greater than --objective-at-most"):
        select_study_report(
            payload,
            objective_at_least=30.0,
            objective_at_most=10.0,
        )


def test_select_study_report_rejects_inverted_duration_thresholds(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)

    with pytest.raises(ValueError, match="--duration-at-least cannot be greater than --duration-at-most"):
        select_study_report(
            payload,
            duration_at_least=30.0,
            duration_at_most=10.0,
        )


def test_select_study_report_groups_failed_trials_by_error_type(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["trials"] = [
        *payload["trials"],
        {
            "trial_index": 3,
            "status": "failed",
            "params": {"total_timesteps": 96, "algo_kwargs.learning_rate": 0.01},
            "objective_value": None,
            "run_dir": None,
            "checkpoint_path": None,
            "error": "RuntimeError: diverged",
            "started_at": "2026-04-02T00:06:00+00:00",
            "ended_at": "2026-04-02T00:07:00+00:00",
        },
    ]

    selected = select_study_report(payload)

    assert selected["selected_error_type_summaries"] == [
        {
            "error_type": "RuntimeError",
            "errors": ["RuntimeError: boom", "RuntimeError: diverged"],
            "failed_trials": 2,
            "selected_trial_share": 0.5,
            "failed_trial_share": 1.0,
            "trial_indices": [1, 3],
        }
    ]


def test_select_study_report_can_focus_parameter_value_summary(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        focus_param="total_timesteps",
    )

    assert selected["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in selected["focused_parameter_value_summary"]] == [128, 32, 64]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "best-objective-value",
    }


def test_select_study_report_can_focus_parameter_value_summary_with_custom_sort(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        focus_param="total_timesteps",
        focus_sort_by="value",
    )

    assert selected["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in selected["focused_parameter_value_summary"]] == [32, 64, 128]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "value",
    }


def test_select_study_report_can_focus_parameter_value_summary_with_incumbent_update_sort(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        focus_param="total_timesteps",
        focus_sort_by="incumbent-updates",
    )

    assert selected["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in selected["focused_parameter_value_summary"]] == [128, 32, 64]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "incumbent-updates",
    }


def test_select_study_report_can_focus_parameter_value_summary_with_duration_sort(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["trials"][0]["ended_at"] = "2026-04-02T00:00:30+00:00"
    payload["trials"][1]["ended_at"] = "2026-04-02T00:03:30+00:00"
    payload["trials"][2]["ended_at"] = "2026-04-02T00:05:00+00:00"
    selected = select_study_report(
        payload,
        focus_param="total_timesteps",
        focus_sort_by="mean-duration-seconds",
    )

    assert selected["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in selected["focused_parameter_value_summary"]] == [32, 128, 64]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "mean-duration-seconds",
    }


def test_select_study_report_can_focus_parameter_value_summary_with_top_k(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(
        payload,
        focus_param="total_timesteps",
        focus_top_k=2,
    )

    assert selected["focused_parameter_name"] == "total_timesteps"
    assert [entry["value"] for entry in selected["focused_parameter_value_summary"]] == [128, 32]
    assert selected["report_filters"] == {
        "status": "all",
        "sort_by": "trial-index",
        "descending": False,
        "top_k": None,
        "focus_param": "total_timesteps",
        "focus_sort_by": "best-objective-value",
        "focus_top_k": 2,
    }


def test_select_study_report_builds_selected_analytics(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    selected = select_study_report(payload)

    assert selected["selected_best_trial_index"] == 2
    assert selected["selected_best_objective_value"] == 30.0
    assert selected["selected_status_counts"] == {"completed": 2, "failed": 1}
    assert selected["selected_objective_summary"] == {
        "completed_trials": 2,
        "failed_trials": 1,
        "min": 10.0,
        "max": 30.0,
        "mean": 20.0,
        "median": 20.0,
    }
    assert selected["selected_error_summaries"] == [
        {
            "error": "RuntimeError: boom",
            "failed_trials": 1,
            "selected_trial_share": 0.3333333333,
            "failed_trial_share": 1.0,
            "trial_indices": [1],
        }
    ]
    assert selected["selected_error_type_summaries"] == [
        {
            "error_type": "RuntimeError",
            "errors": ["RuntimeError: boom"],
            "failed_trials": 1,
            "selected_trial_share": 0.3333333333,
            "failed_trial_share": 1.0,
            "trial_indices": [1],
        }
    ]
    assert selected["selected_duration_summary"] == {
        "timed_trials": 3,
        "untimed_trials": 0,
        "min_seconds": 60.0,
        "max_seconds": 60.0,
        "mean_seconds": 60.0,
        "median_seconds": 60.0,
    }
    assert selected["selected_incumbent_trace"] == [
        {
            "trial_index": 0,
            "status": "completed",
            "objective_value": 10.0,
            "selected_incumbent_trial_index": 0,
            "selected_incumbent_objective_value": 10.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": None,
            "selected_incumbent_trials_since_previous_update": None,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
        {
            "trial_index": 1,
            "status": "failed",
            "objective_value": None,
            "selected_incumbent_trial_index": 0,
            "selected_incumbent_objective_value": 10.0,
            "selected_is_incumbent_update": False,
            "selected_incumbent_update_improvement": None,
            "selected_incumbent_trials_since_previous_update": None,
            "selected_incumbent_age_trials": 1,
            "selected_incumbent_age_seconds": 120.0,
        },
        {
            "trial_index": 2,
            "status": "completed",
            "objective_value": 30.0,
            "selected_incumbent_trial_index": 2,
            "selected_incumbent_objective_value": 30.0,
            "selected_is_incumbent_update": True,
            "selected_incumbent_update_improvement": 20.0,
            "selected_incumbent_trials_since_previous_update": 2,
            "selected_incumbent_age_trials": 0,
            "selected_incumbent_age_seconds": 0.0,
        },
    ]
    assert selected["selected_incumbent_update_summary"] == {
        "incumbent_update_count": 2,
        "first_incumbent_trial_index": 0,
        "latest_incumbent_trial_index": 2,
        "latest_incumbent_objective_value": 30.0,
        "mean_improvement_over_previous": 20.0,
        "max_improvement_over_previous": 20.0,
        "mean_trials_since_previous_update": 2.0,
        "max_trials_since_previous_update": 2,
    }
    assert selected["selected_incumbent_staleness_summary"] == {
        "latest_incumbent_age_trials": 0,
        "latest_incumbent_age_seconds": 0.0,
        "max_incumbent_age_trials": 1,
        "max_incumbent_age_seconds": 120.0,
    }
    assert selected["selected_objective_duration_frontier"] == [
        {
            "trial_index": 2,
            "objective_value": 30.0,
            "duration_seconds": 60.0,
            "selected_best_objective_delta": 0.0,
            "params": {
                "algo_kwargs.learning_rate": 0.0003,
                "total_timesteps": 128,
            },
        }
    ]
    assert [trial["duration_seconds"] for trial in selected["trials"]] == [60.0, 60.0, 60.0]
    assert [trial["selected_incumbent_trial_index"] for trial in selected["trials"]] == [0, 0, 2]
    assert [trial["selected_incumbent_objective_value"] for trial in selected["trials"]] == [10.0, 10.0, 30.0]
    assert [trial["selected_is_incumbent_update"] for trial in selected["trials"]] == [True, False, True]
    assert [trial["selected_incumbent_update_improvement"] for trial in selected["trials"]] == [None, None, 20.0]
    assert [trial["selected_incumbent_trials_since_previous_update"] for trial in selected["trials"]] == [
        None,
        None,
        2,
    ]
    assert [trial["selected_incumbent_age_trials"] for trial in selected["trials"]] == [0, 1, 0]
    assert [trial["selected_incumbent_age_seconds"] for trial in selected["trials"]] == [0.0, 120.0, 0.0]
    assert [trial["is_objective_duration_frontier"] for trial in selected["trials"]] == [False, False, True]
    assert selected["selected_parameter_summaries"]["total_timesteps"] == {
        "completed_unique_values": [32, 128],
        "failed_unique_values": [64],
        "observed_unique_values": [32, 64, 128],
        "observed_unique_count": 3,
        "search_space_kind": "int",
        "candidate_count": 4,
        "coverage_ratio": 0.75,
        "selected_best_value": 128,
        "numeric_min": 32.0,
        "numeric_max": 128.0,
        "numeric_mean": 80.0,
    }
    assert selected["selected_parameter_summaries"]["algo_kwargs.learning_rate"]["selected_best_value"] == 0.0003
    assert selected["selected_parameter_summaries"]["algo_kwargs.learning_rate"]["candidate_count"] == 3
    assert selected["selected_parameter_summaries"]["algo_kwargs.learning_rate"]["coverage_ratio"] == 1.0
    assert selected["selected_parameter_incumbent_summaries"] == {
        "algo_kwargs.learning_rate": {
            "incumbent_update_count": 2,
            "contributing_values": [0.0003, 0.001],
            "contributing_value_count": 2,
            "top_incumbent_value": 0.0003,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 0.0003,
            "latest_incumbent_trial_index": 2,
        },
        "total_timesteps": {
            "incumbent_update_count": 2,
            "contributing_values": [32, 128],
            "contributing_value_count": 2,
            "top_incumbent_value": 128,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 128,
            "latest_incumbent_trial_index": 2,
        },
    }
    assert selected["selected_parameter_incumbent_leaderboard"] == [
        {
            "name": "algo_kwargs.learning_rate",
            "incumbent_update_count": 2,
            "contributing_values": [0.0003, 0.001],
            "contributing_value_count": 2,
            "top_incumbent_value": 0.0003,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 0.0003,
            "latest_incumbent_trial_index": 2,
        },
        {
            "name": "total_timesteps",
            "incumbent_update_count": 2,
            "contributing_values": [32, 128],
            "contributing_value_count": 2,
            "top_incumbent_value": 128,
            "top_incumbent_value_updates": 1,
            "latest_incumbent_value": 128,
            "latest_incumbent_trial_index": 2,
        },
    ]
    assert selected["selected_parameter_effect_leaderboard"] == [
        {
            "name": "algo_kwargs.learning_rate",
            "observed_value_count": 3,
            "completed_value_count": 2,
            "best_objective_spread": 20.0,
            "mean_objective_spread": 20.0,
            "top_value_by_best_objective": 0.0003,
            "top_best_objective_value": 30.0,
            "bottom_value_by_best_objective": 0.001,
            "bottom_best_objective_value": 10.0,
            "top_value_by_mean_objective": 0.0003,
            "top_mean_objective_value": 30.0,
            "bottom_value_by_mean_objective": 0.001,
            "bottom_mean_objective_value": 10.0,
        },
        {
            "name": "total_timesteps",
            "observed_value_count": 3,
            "completed_value_count": 2,
            "best_objective_spread": 20.0,
            "mean_objective_spread": 20.0,
            "top_value_by_best_objective": 128,
            "top_best_objective_value": 30.0,
            "bottom_value_by_best_objective": 32,
            "bottom_best_objective_value": 10.0,
            "top_value_by_mean_objective": 128,
            "top_mean_objective_value": 30.0,
            "bottom_value_by_mean_objective": 32,
            "bottom_mean_objective_value": 10.0,
        },
    ]
    assert selected["selected_parameter_value_summaries"]["total_timesteps"] == [
        {
            "value": 32,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 10.0,
            "mean_objective_value": 10.0,
            "median_objective_value": 10.0,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 0,
            "selected_best_objective_delta": 20.0,
            "rank_by_best_objective_value": 2,
            "rank_by_mean_objective_value": 2,
        },
        {
            "value": 64,
            "trial_count": 1,
            "completed_trials": 0,
            "failed_trials": 1,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 0.0,
            "failure_rate": 1.0,
            "best_objective_value": None,
            "mean_objective_value": None,
            "median_objective_value": None,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 0,
            "latest_incumbent_trial_index": None,
            "selected_best_objective_delta": None,
            "rank_by_best_objective_value": None,
            "rank_by_mean_objective_value": None,
        },
        {
            "value": 128,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 30.0,
            "mean_objective_value": 30.0,
            "median_objective_value": 30.0,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 2,
            "selected_best_objective_delta": 0.0,
            "rank_by_best_objective_value": 1,
            "rank_by_mean_objective_value": 1,
        },
    ]
    assert selected["selected_parameter_value_summaries"]["algo_kwargs.learning_rate"] == [
        {
            "value": 0.0003,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 30.0,
            "mean_objective_value": 30.0,
            "median_objective_value": 30.0,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 2,
            "selected_best_objective_delta": 0.0,
            "rank_by_best_objective_value": 1,
            "rank_by_mean_objective_value": 1,
        },
        {
            "value": 0.001,
            "trial_count": 1,
            "completed_trials": 1,
            "failed_trials": 0,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 1.0,
            "failure_rate": 0.0,
            "best_objective_value": 10.0,
            "mean_objective_value": 10.0,
            "median_objective_value": 10.0,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 1,
            "latest_incumbent_trial_index": 0,
            "selected_best_objective_delta": 20.0,
            "rank_by_best_objective_value": 2,
            "rank_by_mean_objective_value": 2,
        },
        {
            "value": 0.01,
            "trial_count": 1,
            "completed_trials": 0,
            "failed_trials": 1,
            "timed_trials": 1,
            "untimed_trials": 0,
            "completion_rate": 0.0,
            "failure_rate": 1.0,
            "best_objective_value": None,
            "mean_objective_value": None,
            "median_objective_value": None,
            "min_duration_seconds": 60.0,
            "max_duration_seconds": 60.0,
            "mean_duration_seconds": 60.0,
            "median_duration_seconds": 60.0,
            "incumbent_updates": 0,
            "latest_incumbent_trial_index": None,
            "selected_best_objective_delta": None,
            "rank_by_best_objective_value": None,
            "rank_by_mean_objective_value": None,
        },
    ]
    assert selected["search_efficiency_summary"] == {
        "selected_trial_count": 3,
        "completed_trials": 2,
        "failed_trials": 1,
        "failure_rate": 0.3333333333,
        "selected_best_trial_index": 2,
        "selected_best_objective_value": 30.0,
        "selected_trials_until_best": 3,
        "selected_trial_share_until_best": 1.0,
        "completed_trials_until_best": 2,
        "completed_trial_share_until_best": 1.0,
        "time_to_best_seconds": 300.0,
        "best_vs_median_delta": 10.0,
        "best_vs_mean_delta": 10.0,
        "lowest_coverage_parameter": {
            "name": "total_timesteps",
            "coverage_ratio": 0.75,
            "candidate_count": 4,
            "observed_unique_count": 3,
        },
        "highest_coverage_parameter": {
            "name": "algo_kwargs.learning_rate",
            "coverage_ratio": 1.0,
            "candidate_count": 3,
            "observed_unique_count": 3,
        },
    }
    assert [trial["selected_best_objective_delta"] for trial in selected["trials"]] == [20.0, None, 0.0]


def test_select_study_report_computes_best_delta_for_min_objective(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = load_study_report(study_dir)
    payload["objective"] = {"metric": "eval_return_mean", "mode": "min"}
    payload["trials"][0]["objective_value"] = 30.0
    payload["trials"][2]["objective_value"] = 10.0
    selected = select_study_report(payload, status="completed")

    assert selected["selected_best_trial_index"] == 2
    assert selected["selected_best_objective_value"] == 10.0
    assert [trial["selected_best_objective_delta"] for trial in selected["trials"]] == [20.0, 0.0]
    assert [trial["selected_incumbent_update_improvement"] for trial in selected["trials"]] == [None, 20.0]
    assert [trial["selected_incumbent_trials_since_previous_update"] for trial in selected["trials"]] == [None, 1]
    assert [trial["selected_incumbent_age_trials"] for trial in selected["trials"]] == [0, 0]
    assert [trial["selected_incumbent_age_seconds"] for trial in selected["trials"]] == [0.0, 0.0]
    assert selected["selected_incumbent_update_summary"] == {
        "incumbent_update_count": 2,
        "first_incumbent_trial_index": 0,
        "latest_incumbent_trial_index": 2,
        "latest_incumbent_objective_value": 10.0,
        "mean_improvement_over_previous": 20.0,
        "max_improvement_over_previous": 20.0,
        "mean_trials_since_previous_update": 1.0,
        "max_trials_since_previous_update": 1,
    }
    assert selected["selected_incumbent_staleness_summary"] == {
        "latest_incumbent_age_trials": 0,
        "latest_incumbent_age_seconds": 0.0,
        "max_incumbent_age_trials": 0,
        "max_incumbent_age_seconds": 0.0,
    }


def test_export_selected_study_configs_writes_yaml_bundle_and_manifest(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)

    payload = select_study_report(
        load_study_report(study_dir),
        status="completed",
        sort_by="objective-value",
        descending=True,
        top_k=1,
    )
    export_dir = tmp_path / "exports"
    summary = export_selected_study_configs(payload, export_dir)

    exported_config_path = export_dir / "rank-001_trial-0002.yaml"
    manifest_path = export_dir / "manifest.json"
    exported_payload = yaml.safe_load(exported_config_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary == {
        "output_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "exported_count": 1,
        "skipped_trial_indices": [],
    }
    assert exported_payload["total_timesteps"] == 128
    assert exported_payload["algo_kwargs"]["learning_rate"] == 0.0003
    assert manifest_payload["exported_trials"][0]["trial_index"] == 2
    assert manifest_payload["exported_trials"][0]["rank"] == 1
    assert manifest_payload["exported_trials"][0]["exported_config_path"] == str(exported_config_path)


def test_export_selected_study_configs_ignores_failed_trials_even_with_configs(tmp_path: Path) -> None:
    study_dir = _write_study_report_fixture(tmp_path)
    failed_run_dir = study_dir / "trials" / "trial-b"
    (failed_run_dir / "config.yaml").write_text(
        json.dumps(
            {
                "total_timesteps": 64,
                "algo_kwargs": {"learning_rate": 0.01},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = select_study_report(load_study_report(study_dir), status="all")
    export_dir = tmp_path / "exports"

    summary = export_selected_study_configs(payload, export_dir)

    assert summary == {
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


def test_run_study_supports_optuna_backend(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-optuna.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_optuna_tune",
                "  backend: optuna",
                "  sampler: random",
                "  num_trials: 2",
                "  seed: 9",
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

    result = run_study(load_study_config(study_config_path))

    study_dir = study_root / "ppo_optuna_tune"
    study_payload = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    assert result.study_dir == study_dir
    assert study_payload["backend"] == "optuna"
    assert study_payload["trial_count"] == 2
    assert study_payload["status_counts"]["completed"] == 2
    assert result.best_trial_index is not None


def test_resume_study_completes_missing_native_trials(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-resume-native.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_resume_native",
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

    study_config = load_study_config(study_config_path)
    run_study(study_config)
    study_dir = study_root / "ppo_resume_native"
    trials_jsonl_path = study_dir / "trials.jsonl"
    first_record = json.loads(trials_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    trials_jsonl_path.write_text(json.dumps(first_record) + "\n", encoding="utf-8")
    (study_dir / "study.json").write_text(
        json.dumps(
            {
                "study_name": "ppo_resume_native",
                "backend": "native",
                "sampler": "grid",
                "objective": {"metric": "global_step", "mode": "max"},
                "base_config_path": str(base_config),
                "output_dir": str(study_root),
                "trial_count": 1,
                "status_counts": {"completed": 1},
                "best_trial_index": 0,
                "best_objective_value": 32.0,
                "best_run_dir": first_record["run_dir"],
                "best_checkpoint_path": first_record["checkpoint_path"],
                "study_config": serialize_study_config(study_config),
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = resume_study(study_dir)

    updated_payload = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    updated_records = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(updated_records) == 2
    assert updated_payload["trial_count"] == 2
    assert updated_payload["status_counts"]["completed"] == 2
    assert result.best_trial_index == 1
    assert result.best_objective_value == 64.0


def test_resume_study_completes_missing_optuna_trials(tmp_path: Path) -> None:
    base_config = _write_base_train_config(tmp_path)
    study_config_path = tmp_path / "study-resume-optuna.yaml"
    study_root = tmp_path / "studies"
    study_config_path.write_text(
        "\n".join(
            [
                f"base_config: {base_config}",
                f"output_dir: {study_root}",
                "study:",
                "  name: ppo_resume_optuna",
                "  backend: optuna",
                "  sampler: random",
                "  num_trials: 3",
                "  seed: 4",
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

    study_config = load_study_config(study_config_path)
    run_study(study_config)
    study_dir = study_root / "ppo_resume_optuna"
    trials_jsonl_path = study_dir / "trials.jsonl"
    first_record = json.loads(trials_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    trials_jsonl_path.write_text(json.dumps(first_record) + "\n", encoding="utf-8")
    (study_dir / "study.json").write_text(
        json.dumps(
            {
                "study_name": "ppo_resume_optuna",
                "backend": "optuna",
                "sampler": "random",
                "objective": {"metric": "global_step", "mode": "max"},
                "base_config_path": str(base_config),
                "output_dir": str(study_root),
                "trial_count": 1,
                "status_counts": {"completed": 1},
                "best_trial_index": 0,
                "best_objective_value": float(first_record["objective_value"]),
                "best_run_dir": first_record["run_dir"],
                "best_checkpoint_path": first_record["checkpoint_path"],
                "study_config": serialize_study_config(study_config),
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = resume_study(study_dir)

    updated_payload = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    updated_records = [
        json.loads(line)
        for line in (study_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(updated_records) == 3
    assert updated_payload["trial_count"] == 3
    assert updated_payload["status_counts"]["completed"] == 3
    assert result.best_trial_index is not None
