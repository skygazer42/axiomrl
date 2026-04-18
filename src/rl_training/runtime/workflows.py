from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from rl_training.cli_config import load_config
from rl_training.experiment.checkpointing import CheckpointState, load_checkpoint
from rl_training.experiment.benchmarking import augment_metrics_with_benchmark
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.run_utils import resolve_device
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict

_CHECKPOINT_PATH_REQUIRED_ERROR = "checkpoint_path must not be None"


def _config_from_payload(payload: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        algo=payload["algo"],
        env_id=payload["env_id"],
        seed=int(payload["seed"]),
        total_timesteps=int(payload["total_timesteps"]),
        output_dir=Path(payload["output_dir"]),
        execution_backend=payload.get("execution_backend", "local_sync"),
        device=payload.get("device", "auto"),
        num_envs=int(payload.get("num_envs", 1)),
        eval_episodes=int(payload.get("eval_episodes", 5)),
        log_interval=int(payload.get("log_interval", 1)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 1)),
        tags=tuple(payload.get("tags", ())),
        benchmark=dict(payload.get("benchmark", {})),
        algo_kwargs=dict(payload.get("algo_kwargs", {})),
        env_kwargs=dict(payload.get("env_kwargs", {})),
    )


def _resolve_resume_config(
    checkpoint_state: CheckpointState,
    *,
    config_path: str | Path | None = None,
) -> TrainConfig:
    checkpoint_config = _config_from_payload(checkpoint_state.config)
    if config_path is None:
        return checkpoint_config

    resume_config = load_config(config_path)
    mismatches: list[str] = []
    if resume_config.algo != checkpoint_config.algo:
        mismatches.append(f"algo={resume_config.algo!r} expected {checkpoint_config.algo!r}")
    if resume_config.env_id != checkpoint_config.env_id:
        mismatches.append(f"env_id={resume_config.env_id!r} expected {checkpoint_config.env_id!r}")
    if resume_config.seed != checkpoint_config.seed:
        mismatches.append(f"seed={resume_config.seed!r} expected {checkpoint_config.seed!r}")
    if resume_config.num_envs != checkpoint_config.num_envs:
        mismatches.append(f"num_envs={resume_config.num_envs!r} expected {checkpoint_config.num_envs!r}")
    if mismatches:
        resolved_path = Path(config_path).resolve()
        raise ValueError(
            f"resume config {resolved_path} is incompatible with checkpoint: " + ", ".join(mismatches)
        )
    return resume_config


def _prepare_checkpoint_for_resume(
    checkpoint_state: CheckpointState,
    *,
    config: TrainConfig,
) -> CheckpointState:
    checkpoint_config = _config_from_payload(checkpoint_state.config)
    if checkpoint_config.env_kwargs == config.env_kwargs:
        return checkpoint_state

    trainer_state = dict(checkpoint_state.trainer_state)
    trainer_state["resume_context"] = {}
    metadata = dict(checkpoint_state.metadata)
    metadata["resume_replay_reset_reason"] = "env_kwargs_changed"
    return CheckpointState(
        algorithm_state=checkpoint_state.algorithm_state,
        buffer_state=None,
        trainer_state=trainer_state,
        config=checkpoint_state.config,
        metadata=metadata,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    num_episodes: int | None = None,
    device: str = "auto",
) -> MetricDict:
    if checkpoint_path is None:
        raise ValueError(_CHECKPOINT_PATH_REQUIRED_ERROR)

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _config_from_payload(checkpoint_state.config)
    if num_episodes is not None:
        config = replace(config, eval_episodes=num_episodes)

    resolved_device = resolve_device(device if device != "auto" else config.device)
    spec = get_algorithm_spec(config.algo)
    metrics = spec.evaluate_fn(config, checkpoint_state, resolved_device, config.eval_episodes)
    return augment_metrics_with_benchmark(metrics, config.benchmark)


def predict_checkpoint(
    checkpoint_path: str | Path | None,
    obs: object,
    *,
    deterministic: bool = True,
    device: str = "auto",
) -> int | np.ndarray:
    if checkpoint_path is None:
        raise ValueError(_CHECKPOINT_PATH_REQUIRED_ERROR)

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _config_from_payload(checkpoint_state.config)
    resolved_device = resolve_device(device if device != "auto" else config.device)
    spec = get_algorithm_spec(config.algo)
    return spec.predict_fn(config, checkpoint_state, resolved_device, obs, deterministic)


def resume_training(
    checkpoint_path: str | Path | None,
    *,
    config_path: str | Path | None = None,
    total_timesteps: int | None = None,
    output_dir: str | Path | None = None,
    execution_backend: str | None = None,
    eval_episodes: int | None = None,
    run_suffix: str | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if checkpoint_path is None:
        raise ValueError(_CHECKPOINT_PATH_REQUIRED_ERROR)

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _resolve_resume_config(checkpoint_state, config_path=config_path)
    checkpoint_state = _prepare_checkpoint_for_resume(checkpoint_state, config=config)

    overrides: dict[str, Any] = {}
    if total_timesteps is not None:
        overrides["total_timesteps"] = int(total_timesteps)
    if output_dir is not None:
        overrides["output_dir"] = Path(output_dir)
    if execution_backend is not None:
        overrides["execution_backend"] = str(execution_backend)
    if eval_episodes is not None:
        overrides["eval_episodes"] = int(eval_episodes)
    if overrides:
        config = replace(config, **overrides)
    spec = get_algorithm_spec(config.algo)
    return spec.train_fn(
        config,
        run_suffix=run_suffix,
        checkpoint_state=checkpoint_state,
        callbacks=callbacks,
    )
