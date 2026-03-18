from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from rl_training.experiment.benchmarking import augment_metrics_with_benchmark, resolve_best_checkpoint_config
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
                "benchmark": config.benchmark,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_run_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _is_better_metric(candidate: float, incumbent: float | None, *, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    return candidate > incumbent


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
    augment_metrics_with_benchmark(metrics, config.benchmark)
    global_step = int(trainer_state.get("global_step", 0))
    checkpoint_path = run_context.checkpoints_dir / f"step_{global_step}.pt"
    metadata_payload = _load_run_metadata(run_context.metadata_path)
    best_settings = resolve_best_checkpoint_config(config.benchmark)
    best_info = metadata_payload.get("best_checkpoint")
    if not isinstance(best_info, dict):
        best_info = None

    best_metric_value: float | None = None
    if best_info is not None and "metric_value" in best_info:
        best_metric_value = float(best_info["metric_value"])

    current_metric_value: float | None = None
    if best_settings.metric_name in metrics:
        current_metric_value = float(metrics[best_settings.metric_name])

    should_update_best = (
        current_metric_value is not None
        and _is_better_metric(current_metric_value, best_metric_value, mode=best_settings.metric_mode)
    )

    if should_update_best:
        best_checkpoint_alias = run_context.checkpoints_dir / "best.pt"
        best_info = {
            "path": str(best_checkpoint_alias),
            "source_checkpoint_path": str(checkpoint_path),
            "metric_name": best_settings.metric_name,
            "metric_mode": best_settings.metric_mode,
            "metric_value": float(current_metric_value),
            "global_step": global_step,
        }
        if "eval_human_normalized_score" in metrics:
            best_info["eval_human_normalized_score"] = float(metrics["eval_human_normalized_score"])

    if best_info is not None:
        metrics["best_checkpoint_path"] = str(best_info["path"])
        metrics[f"best_{best_settings.metric_name}"] = float(best_info["metric_value"])
        if "eval_human_normalized_score" in best_info:
            metrics["best_eval_human_normalized_score"] = float(best_info["eval_human_normalized_score"])

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
                "best_checkpoint": best_info,
            },
        ),
    )

    if should_update_best and best_info is not None:
        shutil.copy2(checkpoint_path, Path(best_info["path"]))

    metadata_payload["latest_checkpoint_path"] = str(checkpoint_path)
    metadata_payload["latest_metrics"] = dict(metrics)
    if best_info is not None:
        metadata_payload["best_checkpoint"] = best_info
    run_context.metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return checkpoint_path
