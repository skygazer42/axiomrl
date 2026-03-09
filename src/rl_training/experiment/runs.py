from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rl_training.experiment.config import TrainConfig


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    config_path: Path
    metadata_path: Path


def create_run_context(config: TrainConfig, run_suffix: str | None = None) -> RunContext:
    suffix = run_suffix or datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"{config.algo}__{config.env_id}__seed{config.seed}__{suffix}"
    run_dir = config.output_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tensorboard_dir=tensorboard_dir,
        config_path=run_dir / "config.yaml",
        metadata_path=run_dir / "metadata.json",
    )
