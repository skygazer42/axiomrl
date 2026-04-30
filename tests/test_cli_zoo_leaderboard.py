import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiomrl.cli import main


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
    assert summary["top_movers_by_normalized_delta"][1][
        "delta_vs_baseline_latest_eval_human_normalized_score_mean"
    ] == (pytest.approx(5.0))
