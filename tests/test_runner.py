import json
from pathlib import Path

import pytest

from rl_training.experiment.sweeps import SeedSweepPlan
from rl_training.runtime import runner as runner_module
from rl_training.runtime.runner import FunctionRunner
from rl_training.runtime.trainer import TrainResult


def test_function_runner_invokes_run_fn(tmp_path: Path) -> None:
    expected = TrainResult(
        run_dir=tmp_path,
        checkpoint_path=None,
        metrics={"global_step": 4.0},
    )
    runner = FunctionRunner(run_fn=lambda: expected)

    result = runner.run()

    assert result is expected


def test_benchmark_runner_aggregates_child_runs(tmp_path: Path) -> None:
    benchmark_runner_cls = getattr(runner_module, "BenchmarkRunner", None)
    assert benchmark_runner_cls is not None

    outputs = {
        3: TrainResult(
            run_dir=tmp_path / "run-seed3",
            checkpoint_path=tmp_path / "run-seed3" / "checkpoint.pt",
            metrics={"eval_return_mean": 10.0, "global_step": 64.0},
        ),
        5: TrainResult(
            run_dir=tmp_path / "run-seed5",
            checkpoint_path=tmp_path / "run-seed5" / "checkpoint.pt",
            metrics={"eval_return_mean": 14.0, "global_step": 64.0},
        ),
    }
    for result in outputs.values():
        result.run_dir.mkdir(parents=True, exist_ok=True)
        assert result.checkpoint_path is not None
        result.checkpoint_path.write_text("checkpoint", encoding="utf-8")

    runner = benchmark_runner_cls(
        seed_sweep=SeedSweepPlan(seeds=(3, 5)),
        make_runner=lambda seed: FunctionRunner(run_fn=lambda: outputs[seed]),
        summary_path=tmp_path / "benchmark-summary.json",
    )

    aggregate = runner.run()

    assert aggregate.metrics["benchmark_run_count"] == 2.0
    assert aggregate.metrics["eval_return_mean_mean"] == 12.0
    assert "eval_return_mean_std" in aggregate.metrics
    assert isinstance(aggregate.metrics["eval_return_mean_std"], float)
    assert aggregate.metrics["eval_return_mean_std"] >= 0.0
    assert aggregate.metrics["global_step_mean"] == 64.0
    assert aggregate.benchmark_summary_path == tmp_path / "benchmark-summary.json"

    summary_payload = json.loads((tmp_path / "benchmark-summary.json").read_text(encoding="utf-8"))
    assert summary_payload["aggregate_metrics"]["eval_return_mean_mean"] == 12.0

    runs_by_seed = {entry["seed"]: entry for entry in summary_payload["runs"]}
    assert set(runs_by_seed) == {3, 5}
    assert runs_by_seed[3]["run_dir"] == str(outputs[3].run_dir)
    assert runs_by_seed[3]["checkpoint_path"] == str(outputs[3].checkpoint_path)
    assert runs_by_seed[3]["metrics"]["eval_return_mean"] == 10.0
    assert runs_by_seed[5]["run_dir"] == str(outputs[5].run_dir)
    assert runs_by_seed[5]["checkpoint_path"] == str(outputs[5].checkpoint_path)
    assert runs_by_seed[5]["metrics"]["eval_return_mean"] == 14.0


def test_benchmark_runner_rejects_existing_summary_path(tmp_path: Path) -> None:
    benchmark_runner_cls = getattr(runner_module, "BenchmarkRunner", None)
    assert benchmark_runner_cls is not None

    run_dir = tmp_path / "run-seed3"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tmp_path / "benchmark-summary.json"
    summary_path.write_text('{"status": "preexisting"}', encoding="utf-8")

    runner = benchmark_runner_cls(
        seed_sweep=SeedSweepPlan(seeds=(3,)),
        make_runner=lambda seed: FunctionRunner(
            run_fn=lambda: TrainResult(
                run_dir=run_dir,
                checkpoint_path=None,
                metrics={"eval_return_mean": float(seed), "global_step": 64.0},
            )
        ),
        summary_path=summary_path,
    )

    with pytest.raises(FileExistsError, match="benchmark summary already exists"):
        runner.run()

    assert summary_path.read_text(encoding="utf-8") == '{"status": "preexisting"}'
