from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from rl_training.experiment.config import TrainConfig


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    config_path: Path
    metadata_path: Path


_RUN_ID_UNSAFE_CHARS = re.compile(r"[\\\\/:\s]+")


def _sanitize_run_id_component(value: str) -> str:
    sanitized = _RUN_ID_UNSAFE_CHARS.sub("-", value).strip("-")
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized or "unknown"


def create_run_context(config: TrainConfig, run_suffix: str | None = None) -> RunContext:
    attempt = 0
    algo_component = _sanitize_run_id_component(config.algo)
    env_component = _sanitize_run_id_component(config.env_id)
    while True:
        if run_suffix is not None:
            suffix = run_suffix
        else:
            base_suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            suffix = base_suffix if attempt == 0 else f"{base_suffix}-{attempt}"

        run_id = f"{algo_component}__{env_component}__seed{config.seed}__{suffix}"
        run_dir = config.output_dir / run_id
        checkpoints_dir = run_dir / "checkpoints"
        tensorboard_dir = run_dir / "tensorboard"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if run_suffix is not None:
                raise
            attempt += 1

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
