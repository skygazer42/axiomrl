from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.runtime.callbacks import CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.resume_state import (
    capture_global_random_state,
    capture_vector_env_resume_state,
    restore_global_random_state,
    restore_vector_env_resume_state,
)
from rl_training.runtime.trainer import TrainerState
from rl_training.runtime.types import MetricDict


def store_vector_transitions(
    replay_buffer: ReplayBuffer,
    *,
    obs: np.ndarray,
    actions: Any,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        replay_buffer.add(
            obs=obs[env_index],
            actions=actions[env_index],
            rewards=float(rewards[env_index]),
            next_obs=next_obs[env_index],
            dones=float(dones[env_index]),
        )


def emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    num_envs: int,
    dones: np.ndarray,
    replay_buffer: ReplayBuffer,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=num_envs,
            num_episodes=int(np.sum(dones)),
            metrics={"global_step": float(global_step), "buffer_size": float(len(replay_buffer))},
            last_obs=obs,
        ),
    )


def maybe_update_algorithm(
    *,
    algorithm: object,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    global_step: int,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int]:
    if len(replay_buffer) < max(batch_size, learning_starts):
        return latest_update_metrics, update_count
    if global_step % train_frequency != 0:
        return latest_update_metrics, update_count

    result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)  # type: ignore[attr-defined]
    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps


def build_replay_metrics(
    latest_update_metrics: MetricDict,
    *,
    global_step: int,
    replay_buffer: ReplayBuffer,
    update_count: int,
    extra_metrics: MetricDict | None = None,
) -> MetricDict:
    return {
        **latest_update_metrics,
        **(extra_metrics or {}),
        "global_step": float(global_step),
        "buffer_size": float(len(replay_buffer)),
        "gradient_steps": float(update_count),
    }


def maybe_run_evaluation(
    *,
    should_run_eval: bool,
    algorithm: object,
    evaluate: Callable[[], MetricDict],
    logger: Any,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    metrics: MetricDict,
    global_step: int,
) -> tuple[MetricDict, bool]:
    if not should_run_eval:
        return metrics, False

    algorithm.set_eval_mode()  # type: ignore[attr-defined]
    eval_metrics = evaluate()
    algorithm.set_train_mode()  # type: ignore[attr-defined]

    merged_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(merged_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, merged_metrics)
    return merged_metrics, trainer_state.should_stop


def restore_replay_training_state(
    *,
    algorithm: object,
    replay_buffer: ReplayBuffer,
    envs: gym.vector.VectorEnv,
    checkpoint_state: CheckpointState | None,
) -> tuple[object | None, int, int]:
    if checkpoint_state is None:
        return None, 0, 0

    algorithm.load_state_dict(checkpoint_state.algorithm_state)  # type: ignore[attr-defined]
    if checkpoint_state.buffer_state is not None:
        replay_buffer.load_state_dict(checkpoint_state.buffer_state)

    restored_obs = None
    resume_context = checkpoint_state.trainer_state.get("resume_context")
    if isinstance(resume_context, dict):
        env_resume_state = resume_context.get("env_state")
        if isinstance(env_resume_state, dict):
            restored_obs = restore_vector_env_resume_state(envs, env_resume_state)
        random_state = resume_context.get("random_state")
        if isinstance(random_state, dict):
            restore_global_random_state(random_state)

    return (
        restored_obs,
        int(checkpoint_state.trainer_state.get("global_step", 0)),
        int(checkpoint_state.trainer_state.get("update_count", 0)),
    )


def capture_replay_resume_context(envs: gym.vector.VectorEnv) -> dict[str, object]:
    return {
        "env_state": capture_vector_env_resume_state(envs),
        "random_state": capture_global_random_state(),
    }
