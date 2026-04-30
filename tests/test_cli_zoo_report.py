import csv
import io
import json
from pathlib import Path

import pytest

from axiomrl.cli import main


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


def test_report_subcommand_supports_nested_run_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_dir = tmp_path / "runs"
    nested_parent = runs_dir / "ppo__ALE"

    first_run_dir = nested_parent / "Breakout-v5__seed9__demo"
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

    second_run_dir = nested_parent / "Breakout-v5__seed11__demo"
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
    assert "ppo__ALE/Breakout-v5__seed9__demo" in captured.out
    assert "ppo__ALE/Breakout-v5__seed11__demo" in captured.out
    assert "aggregate algo=ppo env_id=ALE/Breakout-v5 runs=2 seeds=9,11" in captured.out


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
