from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from rl_training.experiment.checkpointing import CheckpointState, save_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.logging import RunLogger
from rl_training.experiment.runs import RunContext, create_run_context
from rl_training.runtime.types import MetricDict


@dataclass(slots=True)
class RunArtifacts:
    run_context: RunContext
    logger: RunLogger

    def close(self) -> None:
        self.logger.close()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            try:
                probe = torch.empty(1, device="cuda")
                del probe
                return torch.device("cuda")
            except RuntimeError:
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device(device)


def serialize_train_config(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["tags"] = list(config.tags)
    return payload


def _write_run_files(config: TrainConfig, *, run_context: RunContext) -> None:
    config_payload = serialize_train_config(config)
    run_context.config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    run_context.metadata_path.write_text(
        json.dumps(
            {
                "algo": config.algo,
                "env_id": config.env_id,
                "seed": config.seed,
                "output_dir": str(run_context.run_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_training_run(config: TrainConfig, *, run_suffix: str | None = None) -> RunArtifacts:
    run_context = create_run_context(config, run_suffix=run_suffix)
    logger = RunLogger(run_context.run_dir, tensorboard_dir=run_context.tensorboard_dir)
    _write_run_files(config, run_context=run_context)
    logger.log_config(serialize_train_config(config))
    return RunArtifacts(run_context=run_context, logger=logger)


def save_training_checkpoint(
    *,
    run_context: RunContext,
    config: TrainConfig,
    algorithm_state: dict[str, Any],
    buffer_state: dict[str, Any] | None,
    trainer_state: dict[str, Any],
    metrics: MetricDict,
) -> Path:
    global_step = int(trainer_state.get("global_step", 0))
    checkpoint_path = run_context.checkpoints_dir / f"step_{global_step}.pt"
    save_checkpoint(
        checkpoint_path,
        CheckpointState(
            algorithm_state=algorithm_state,
            buffer_state=buffer_state,
            trainer_state=trainer_state,
            config=serialize_train_config(config),
            metadata={
                "run_id": run_context.run_id,
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
            },
        ),
    )
    return checkpoint_path
