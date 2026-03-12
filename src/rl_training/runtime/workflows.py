from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from rl_training.experiment.checkpointing import CheckpointState, load_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.run_utils import resolve_device
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _config_from_payload(payload: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        algo=payload["algo"],
        env_id=payload["env_id"],
        seed=int(payload["seed"]),
        total_timesteps=int(payload["total_timesteps"]),
        output_dir=Path(payload["output_dir"]),
        device=payload.get("device", "auto"),
        num_envs=int(payload.get("num_envs", 1)),
        eval_episodes=int(payload.get("eval_episodes", 5)),
        log_interval=int(payload.get("log_interval", 1)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 1)),
        tags=tuple(payload.get("tags", ())),
        algo_kwargs=dict(payload.get("algo_kwargs", {})),
        env_kwargs=dict(payload.get("env_kwargs", {})),
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    num_episodes: int | None = None,
    device: str = "auto",
) -> MetricDict:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must not be None")

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _config_from_payload(checkpoint_state.config)
    if num_episodes is not None:
        config = replace(config, eval_episodes=num_episodes)

    resolved_device = resolve_device(device if device != "auto" else config.device)
    spec = get_algorithm_spec(config.algo)
    return spec.evaluate_fn(config, checkpoint_state, resolved_device, config.eval_episodes)


def predict_checkpoint(
    checkpoint_path: str | Path | None,
    obs: object,
    *,
    deterministic: bool = True,
    device: str = "auto",
) -> int | np.ndarray:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must not be None")

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _config_from_payload(checkpoint_state.config)
    resolved_device = resolve_device(device if device != "auto" else config.device)
    spec = get_algorithm_spec(config.algo)
    return spec.predict_fn(config, checkpoint_state, resolved_device, obs, deterministic)


def resume_training(
    checkpoint_path: str | Path | None,
    *,
    total_timesteps: int | None = None,
    output_dir: str | Path | None = None,
    eval_episodes: int | None = None,
    run_suffix: str | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must not be None")

    checkpoint_state = load_checkpoint(Path(checkpoint_path))
    config = _config_from_payload(checkpoint_state.config)

    overrides: dict[str, Any] = {}
    if total_timesteps is not None:
        overrides["total_timesteps"] = int(total_timesteps)
    if output_dir is not None:
        overrides["output_dir"] = Path(output_dir)
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
