from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
import warnings

from torch.utils.tensorboard import SummaryWriter

from rl_training.runtime.types import MetricDict


class Logger(Protocol):
    def log_metrics(self, metrics: MetricDict, *, step: int) -> None:
        ...

    def log_config(self, config: dict) -> None:
        ...

    def close(self) -> None:
        ...


class RunLogger:
    """Concrete run logger for JSON artifacts and TensorBoard event files."""

    def __init__(self, run_dir: Path, *, tensorboard_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.config_log_path = run_dir / "config.json"
        self.tensorboard_dir = tensorboard_dir or run_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

    def log_metrics(self, metrics: MetricDict, *, step: int) -> None:
        payload = {"step": step, "metrics": metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, global_step=step)

    def log_config(self, config: dict) -> None:
        self.config_log_path.write_text(
            json.dumps(config, indent=2, default=str),
            encoding="utf-8",
        )
        self.writer.add_text("config", json.dumps(config, indent=2, default=str), global_step=0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class JsonlLogger(RunLogger):
    """Deprecated compatibility wrapper for older imports."""

    def __init__(self, run_dir: Path, *, tensorboard_dir: Path | None = None) -> None:
        warnings.warn(
            "JsonlLogger is deprecated; use RunLogger instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(run_dir, tensorboard_dir=tensorboard_dir)
