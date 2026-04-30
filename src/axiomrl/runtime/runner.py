from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axiomrl.experiment.benchmarking import aggregate_numeric_metrics
from axiomrl.experiment.sweeps import BenchmarkRunRecord, SeedSweepPlan
from axiomrl.runtime.trainer import TrainResult


class Runner(Protocol):
    def run(self) -> TrainResult: ...


@dataclass(slots=True)
class FunctionRunner(Runner):
    run_fn: Callable[[], TrainResult]

    def run(self) -> TrainResult:
        return self.run_fn()


@dataclass(slots=True)
class BenchmarkRunner(Runner):
    seed_sweep: SeedSweepPlan
    make_runner: Callable[[int], Runner]
    summary_path: Path

    def run(self) -> TrainResult:
        if self.summary_path.exists():
            raise FileExistsError(f"benchmark summary already exists: {self.summary_path}")

        records: list[BenchmarkRunRecord] = []
        for seed in self.seed_sweep.seeds:
            child_result = self.make_runner(seed).run()
            records.append(
                BenchmarkRunRecord(
                    seed=seed,
                    run_dir=child_result.run_dir,
                    checkpoint_path=child_result.checkpoint_path,
                    metrics=dict(child_result.metrics),
                )
            )

        aggregate_metrics = aggregate_numeric_metrics([record.metrics for record in records])
        aggregate_metrics["benchmark_run_count"] = float(len(records))

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregate_metrics": aggregate_metrics,
            "runs": [
                {
                    "seed": record.seed,
                    "run_dir": str(record.run_dir),
                    "checkpoint_path": None if record.checkpoint_path is None else str(record.checkpoint_path),
                    "metrics": record.metrics,
                }
                for record in records
            ],
        }
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return TrainResult(
            run_dir=self.summary_path.parent,
            checkpoint_path=None,
            metrics=dict(aggregate_metrics),
            benchmark_summary_path=self.summary_path,
        )
