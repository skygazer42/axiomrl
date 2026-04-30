from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiomrl.experiment.benchmarking import aggregate_numeric_metrics, resolve_score_normalization_config
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.logging import RunLogger
from axiomrl.runtime.run_utils import create_training_run, save_training_checkpoint


def test_score_normalization_source_resolves_named_reference() -> None:
    config = resolve_score_normalization_config(
        {
            "score_normalization": {
                "type": "human_random",
                "source": "atari_breakout_reference",
            }
        }
    )

    assert config is not None
    assert config.strategy == "human_random"
    assert config.random_score == pytest.approx(1.7)
    assert config.human_score == pytest.approx(30.5)


def test_save_training_checkpoint_tracks_best_checkpoint_and_normalized_score(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=32,
        output_dir=tmp_path,
        benchmark={
            "best_metric": "eval_return_mean",
            "best_metric_mode": "max",
            "score_normalization": {
                "type": "human_random",
                "random_score": 0.0,
                "human_score": 200.0,
            },
        },
    )

    artifacts = create_training_run(config, run_suffix="benchmark")
    try:
        first_metrics = {"eval_return_mean": 50.0}
        first_checkpoint = save_training_checkpoint(
            run_context=artifacts.run_context,
            config=config,
            algorithm_state={"weights": [1.0]},
            buffer_state=None,
            trainer_state={"global_step": 10},
            metrics=first_metrics,
        )

        second_metrics = {"eval_return_mean": 80.0}
        second_checkpoint = save_training_checkpoint(
            run_context=artifacts.run_context,
            config=config,
            algorithm_state={"weights": [2.0]},
            buffer_state=None,
            trainer_state={"global_step": 20},
            metrics=second_metrics,
        )
    finally:
        artifacts.close()

    best_checkpoint_path = artifacts.run_context.checkpoints_dir / "best.pt"
    metadata = json.loads(artifacts.run_context.metadata_path.read_text(encoding="utf-8"))

    assert first_checkpoint.exists()
    assert second_checkpoint.exists()
    assert best_checkpoint_path.exists()
    assert first_metrics["eval_human_normalized_score"] == pytest.approx(25.0)
    assert second_metrics["eval_human_normalized_score"] == pytest.approx(40.0)
    assert second_metrics["best_eval_return_mean"] == pytest.approx(80.0)
    assert Path(second_metrics["best_checkpoint_path"]) == best_checkpoint_path
    assert metadata["best_checkpoint"]["metric_name"] == "eval_return_mean"
    assert metadata["best_checkpoint"]["metric_value"] == pytest.approx(80.0)
    assert Path(metadata["best_checkpoint"]["source_checkpoint_path"]) == second_checkpoint
    assert Path(metadata["best_checkpoint"]["path"]) == best_checkpoint_path


def test_run_logger_accepts_benchmark_augmented_metrics(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path)
    try:
        logger.log_metrics(
            {
                "eval_return_mean": 50.0,
                "eval_human_normalized_score": 25.0,
                "best_eval_return_mean": 50.0,
            },
            step=10,
        )
    finally:
        logger.close()

    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.exists()


def test_aggregate_numeric_metrics_uses_population_std_for_shared_numeric_keys() -> None:
    aggregated = aggregate_numeric_metrics(
        [
            {
                "eval_return_mean": 10.0,
                "global_step": 64.0,
                "sample_count": 4,
            },
            {
                "eval_return_mean": 14.0,
                "global_step": 64.0,
                "sample_count": 8,
            },
        ]
    )

    assert aggregated["eval_return_mean_mean"] == pytest.approx(12.0)
    assert aggregated["eval_return_mean_std"] == pytest.approx(2.0)
    assert aggregated["eval_return_mean_min"] == pytest.approx(10.0)
    assert aggregated["eval_return_mean_max"] == pytest.approx(14.0)
    assert aggregated["global_step_mean"] == pytest.approx(64.0)
    assert aggregated["global_step_std"] == pytest.approx(0.0)
    assert aggregated["sample_count_mean"] == pytest.approx(6.0)
    assert aggregated["sample_count_std"] == pytest.approx(2.0)


def test_aggregate_numeric_metrics_only_aggregates_shared_numeric_keys() -> None:
    aggregated = aggregate_numeric_metrics(
        [
            {
                "eval_return_mean": 10.0,
                "global_step": 64.0,
                "status": "ok",
            },
            {
                "eval_return_mean": 14.0,
                "global_step": 64.0,
                "status": "ok",
                "extra_only_here": 1.0,
            },
        ]
    )

    assert "eval_return_mean_mean" in aggregated
    assert "global_step_mean" in aggregated
    assert "status_mean" not in aggregated
    assert "extra_only_here_mean" not in aggregated


def test_aggregate_numeric_metrics_handles_empty_input() -> None:
    assert aggregate_numeric_metrics([]) == {}
