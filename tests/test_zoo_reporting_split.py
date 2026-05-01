import json
from pathlib import Path

from axiomrl.zoo.reporting_render import csv_report_rows, render_csv_report
from axiomrl.zoo.reporting_runs import filter_run_reports, iter_run_reports
from axiomrl.zoo.reporting_stats import aggregate_run_reports


def test_iter_run_reports_supports_nested_run_directories(tmp_path: Path) -> None:
    nested_run_dir = tmp_path / "suite-a" / "run-001"
    nested_run_dir.mkdir(parents=True)
    (nested_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "seed": 7,
                "benchmark": {
                    "suite": "atari",
                    "preset_name": "ppo_cartpole",
                    "protocol_name": "default",
                },
                "latest_metrics": {
                    "eval_return_mean": 11.0,
                    "eval_human_normalized_score": 0.5,
                    "best_eval_return_mean": 13.0,
                    "best_eval_human_normalized_score": 0.7,
                },
                "best_checkpoint": {
                    "path": "runs/checkpoints/best.pt",
                    "metric_value": 13.0,
                    "eval_human_normalized_score": 0.7,
                },
            }
        ),
        encoding="utf-8",
    )

    reports = iter_run_reports(tmp_path)

    assert len(reports) == 1
    assert reports[0]["run_id"] == "suite-a/run-001"
    assert reports[0]["best_minus_latest_eval_return_mean"] == 2.0


def test_filter_run_reports_filters_on_algo_and_env_id() -> None:
    reports = [
        {"algo": "ppo", "env_id": "CartPole-v1", "run_id": "a"},
        {"algo": "dqn", "env_id": "CartPole-v1", "run_id": "b"},
        {"algo": "ppo", "env_id": "Pendulum-v1", "run_id": "c"},
    ]

    filtered = filter_run_reports(reports, algo="ppo", env_id="CartPole-v1")

    assert filtered == [{"algo": "ppo", "env_id": "CartPole-v1", "run_id": "a"}]


def test_aggregate_run_reports_computes_preset_groups_and_alignment_counts() -> None:
    reports = [
        {
            "run_id": "run-a",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "seed": 1,
            "suite": "classic",
            "preset_name": "ppo_cartpole",
            "protocol_name": "default",
            "latest_eval_return_mean": 10.0,
            "latest_eval_human_normalized_score": 0.4,
            "best_eval_return_mean": 15.0,
            "best_eval_human_normalized_score": 0.8,
            "manifest_alignment_status": "aligned",
            "manifest_preset_known": True,
            "manifest_protocol_matches_manifest": True,
        },
        {
            "run_id": "run-b",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "seed": 2,
            "suite": "classic",
            "preset_name": "ppo_cartpole",
            "protocol_name": "default",
            "latest_eval_return_mean": 14.0,
            "latest_eval_human_normalized_score": 0.6,
            "best_eval_return_mean": 17.0,
            "best_eval_human_normalized_score": 0.9,
            "manifest_alignment_status": "mixed",
            "manifest_preset_known": False,
            "manifest_protocol_matches_manifest": False,
        },
    ]

    aggregates = aggregate_run_reports(reports, group_by="preset")

    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate["group"] == "ppo_cartpole"
    assert aggregate["runs"] == 2
    assert aggregate["seed_count"] == 2
    assert aggregate["latest_eval_return_mean_mean"] == 12.0
    assert aggregate["best_eval_return_mean_max"] == 17.0
    assert aggregate["manifest_alignment_total_runs"] == 2
    assert aggregate["manifest_alignment_aligned_runs"] == 1
    assert aggregate["manifest_alignment_drifted_runs"] == 1
    assert aggregate["manifest_alignment_unknown_preset_runs"] == 1
    assert aggregate["manifest_alignment_protocol_mismatch_runs"] == 1


def test_csv_report_rows_and_renderer_include_run_aggregate_and_summary_rows() -> None:
    payload = {
        "suite": "classic",
        "protocol": "default",
        "score_normalization": "none",
        "runs_dir": "runs",
        "runs": [{"run_id": "run-a", "algo": "ppo", "env_id": "CartPole-v1", "seed": 1}],
        "aggregates": [{"group": "ppo::CartPole-v1", "algo": "ppo", "env_id": "CartPole-v1", "runs": 1}],
        "baseline_preset": "ppo_cartpole",
        "baseline_summary": {
            "baseline_preset": "ppo_cartpole",
            "top_movers_by_return_delta": [{"preset_name": "alt", "algo": "ppo", "env_id": "CartPole-v1"}],
            "top_regressions_by_return_delta": [],
            "top_movers_by_normalized_delta": [],
            "top_regressions_by_normalized_delta": [],
        },
    }

    rows = csv_report_rows(payload)
    rendered = render_csv_report(payload)

    assert [row["kind"] for row in rows] == ["run", "aggregate", "summary"]
    assert rows[-1]["summary_kind"] == "top_movers_by_return_delta"
    assert "kind,summary_kind,summary_rank" in rendered
    assert "run-a" in rendered
    assert "ppo::CartPole-v1" in rendered
