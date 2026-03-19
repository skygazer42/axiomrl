from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.rlpd import RLPD
from rl_training.data.offline_dataset import TransitionDataset
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_periodic_eval,
    stop_reason_for_training_limits,
)
from rl_training.runtime.iql_trainer import _build_offline_dataset
from rl_training.runtime.off_policy_trainer_utils import maybe_run_evaluation, store_vector_transitions
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.sac_trainer import _action_bounds, _evaluate_sac_policy, _infer_spaces, _scale_actions
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


@dataclass
class _RLPDTrainingState:
    global_step: int
    update_count: int
    epoch: int
    pretrain_updates_done: int
    latest_update_metrics: MetricDict
    last_offline_batch_size: float
    last_online_batch_size: float
    last_lr_scale: float
    last_learning_rate: float


@dataclass(frozen=True)
class _RLPDLoopConfig:
    learning_rate_schedule: object
    effective_total_updates: int
    warmup_steps: int
    batch_size: int
    device: torch.device
    alpha: float
    offline_pretrain_updates: int
    offline_batch_ratio: float
    learning_starts: int
    train_frequency: int
    gradient_updates_per_step: int
    max_epochs: int | None
    max_updates: int | None


def _validate_offline_batch_ratio(value: float) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"offline_batch_ratio must be between 0 and 1, got {resolved}")
    return resolved


def _concatenate_transition_batches(batches: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("at least one batch is required")
    if len(batches) == 1:
        return dict(batches[0])

    merged: dict[str, torch.Tensor] = {}
    common_keys = set(batches[0]).intersection(*(set(batch) for batch in batches[1:]))
    for key in common_keys:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    return merged


def _sample_mixed_batch(
    *,
    offline_dataset: TransitionDataset,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    offline_batch_ratio: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    offline_batch_size = min(batch_size, max(0, int(round(batch_size * offline_batch_ratio))))
    online_batch_size = batch_size - offline_batch_size

    if len(replay_buffer) == 0:
        offline_batch_size = batch_size
        online_batch_size = 0

    batches: list[dict[str, torch.Tensor]] = []
    if offline_batch_size > 0:
        batches.append(offline_dataset.sample(offline_batch_size, device=device))
    if online_batch_size > 0:
        batches.append(replay_buffer.sample(online_batch_size))
    return _concatenate_transition_batches(batches), offline_batch_size, online_batch_size


def _resolve_effective_total_rlpd_updates(
    config: TrainConfig,
    *,
    offline_pretrain_updates: int,
    gradient_updates_per_step: int,
) -> int:
    online_collection_steps = max(1, math.ceil(config.total_timesteps / max(config.num_envs, 1)))
    nominal_total_updates = offline_pretrain_updates + online_collection_steps * max(gradient_updates_per_step, 1)
    candidates = [nominal_total_updates]

    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    if max_updates is not None:
        candidates.append(max_updates)
    if max_epochs is not None:
        candidates.append(max_epochs)

    resolved = min(candidates)
    if resolved < 1:
        raise ValueError(f"effective total updates must be >= 1, got {resolved}")
    return resolved


def _restore_rlpd_training_state(
    checkpoint_state: CheckpointState | None,
    *,
    batch_size: int,
    offline_pretrain_updates: int,
    learning_rate: float,
) -> _RLPDTrainingState:
    if checkpoint_state is None:
        return _RLPDTrainingState(
            global_step=0,
            update_count=0,
            epoch=0,
            pretrain_updates_done=0,
            latest_update_metrics={},
            last_offline_batch_size=0.0,
            last_online_batch_size=0.0,
            last_lr_scale=1.0,
            last_learning_rate=learning_rate,
        )

    update_count = int(checkpoint_state.trainer_state.get("update_count", 0))
    pretrain_updates_done = int(
        checkpoint_state.trainer_state.get("pretrain_updates_done", min(update_count, offline_pretrain_updates))
    )
    return _RLPDTrainingState(
        global_step=int(checkpoint_state.trainer_state.get("global_step", 0)),
        update_count=update_count,
        epoch=int(checkpoint_state.trainer_state.get("epoch", update_count)),
        pretrain_updates_done=pretrain_updates_done,
        latest_update_metrics={},
        last_offline_batch_size=float(batch_size if pretrain_updates_done > 0 else 0.0),
        last_online_batch_size=0.0,
        last_lr_scale=1.0,
        last_learning_rate=learning_rate,
    )


def _sync_rlpd_trainer_state(trainer_state: TrainerState, state: _RLPDTrainingState) -> None:
    trainer_state.global_step = state.global_step
    trainer_state.epoch = state.epoch
    trainer_state.update_count = state.update_count


def _build_rlpd_metrics(
    state: _RLPDTrainingState,
    *,
    alpha: float,
    offline_dataset_size: int,
    replay_buffer_size: int,
    offline_pretrain_updates: int,
    offline_batch_ratio: float,
) -> MetricDict:
    return {
        **state.latest_update_metrics,
        "alpha": alpha,
        "global_step": float(state.global_step),
        "epoch": float(state.epoch),
        "update_count": float(state.update_count),
        "gradient_steps": float(state.update_count),
        "offline_dataset_size": float(offline_dataset_size),
        "online_buffer_size": float(replay_buffer_size),
        "buffer_size": float(replay_buffer_size),
        "offline_pretrain_updates": float(offline_pretrain_updates),
        "pretrain_updates_done": float(state.pretrain_updates_done),
        "offline_batch_ratio": float(offline_batch_ratio),
        "offline_batch_size": state.last_offline_batch_size,
        "online_batch_size": state.last_online_batch_size,
        "lr_scale": float(state.last_lr_scale),
        "learning_rate": float(state.last_learning_rate),
    }


def _request_stop_for_limits(
    trainer_state: TrainerState,
    *,
    epoch: int,
    update_count: int,
    max_epochs: int | None,
    max_updates: int | None,
) -> bool:
    stop_reason = stop_reason_for_training_limits(
        epoch=epoch,
        update_count=update_count,
        max_epochs=max_epochs,
        max_updates=max_updates,
    )
    if stop_reason is None:
        return False
    trainer_state.request_stop(stop_reason)
    return True


def _emit_offline_dataset_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    offline_dataset_size: int,
    replay_buffer_size: int,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=offline_dataset_size,
            num_episodes=0,
            metrics={
                "offline_dataset_size": float(offline_dataset_size),
                "online_buffer_size": float(replay_buffer_size),
            },
            last_obs=None,
        ),
    )


def _emit_online_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    dones: np.ndarray,
    offline_dataset_size: int,
    replay_buffer: ReplayBuffer,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=int(dones.shape[0]),
            num_episodes=int(np.sum(dones)),
            metrics={
                "global_step": float(global_step),
                "offline_dataset_size": float(offline_dataset_size),
                "online_buffer_size": float(len(replay_buffer)),
            },
            last_obs=obs,
        ),
    )


def _run_rlpd_pretraining(
    *,
    algorithm: RLPD,
    offline_dataset: TransitionDataset,
    replay_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    loop_config: _RLPDLoopConfig,
    state: _RLPDTrainingState,
) -> tuple[_RLPDTrainingState, MetricDict]:
    metrics = _build_rlpd_metrics(
        state,
        alpha=loop_config.alpha,
        offline_dataset_size=len(offline_dataset),
        replay_buffer_size=len(replay_buffer),
        offline_pretrain_updates=loop_config.offline_pretrain_updates,
        offline_batch_ratio=loop_config.offline_batch_ratio,
    )
    while state.pretrain_updates_done < loop_config.offline_pretrain_updates and not trainer_state.should_stop:
        state.last_lr_scale = resolve_schedule_value(
            loop_config.learning_rate_schedule,
            step=state.update_count,
            total_steps=loop_config.effective_total_updates,
            warmup_steps=loop_config.warmup_steps,
        )
        state.last_learning_rate = apply_learning_rate_scale(algorithm, scale=state.last_lr_scale)
        result = algorithm.update(
            offline_dataset.sample(loop_config.batch_size, device=loop_config.device),
            global_step=state.global_step,
        )
        state.pretrain_updates_done += 1
        state.update_count += result.num_gradient_steps
        state.epoch += 1
        state.latest_update_metrics = result.metrics
        state.last_offline_batch_size = float(loop_config.batch_size)
        state.last_online_batch_size = 0.0
        _sync_rlpd_trainer_state(trainer_state, state)
        callback_list.on_update_end(trainer_state, result)

        metrics = _build_rlpd_metrics(
            state,
            alpha=loop_config.alpha,
            offline_dataset_size=len(offline_dataset),
            replay_buffer_size=len(replay_buffer),
            offline_pretrain_updates=loop_config.offline_pretrain_updates,
            offline_batch_ratio=loop_config.offline_batch_ratio,
        )
        if _request_stop_for_limits(
            trainer_state,
            epoch=state.epoch,
            update_count=state.update_count,
            max_epochs=loop_config.max_epochs,
            max_updates=loop_config.max_updates,
        ):
            break
    return state, metrics


def _run_rlpd_online_updates(
    *,
    algorithm: RLPD,
    offline_dataset: TransitionDataset,
    replay_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    loop_config: _RLPDLoopConfig,
    global_step: int,
    state: _RLPDTrainingState,
) -> _RLPDTrainingState:
    if len(replay_buffer) < loop_config.learning_starts or global_step % loop_config.train_frequency != 0:
        return state

    for _ in range(loop_config.gradient_updates_per_step):
        state.last_lr_scale = resolve_schedule_value(
            loop_config.learning_rate_schedule,
            step=state.update_count,
            total_steps=loop_config.effective_total_updates,
            warmup_steps=loop_config.warmup_steps,
        )
        state.last_learning_rate = apply_learning_rate_scale(algorithm, scale=state.last_lr_scale)
        batch, offline_batch_size, online_batch_size = _sample_mixed_batch(
            offline_dataset=offline_dataset,
            replay_buffer=replay_buffer,
            batch_size=loop_config.batch_size,
            offline_batch_ratio=loop_config.offline_batch_ratio,
            device=loop_config.device,
        )
        result = algorithm.update(batch, global_step=global_step)
        state.latest_update_metrics = result.metrics
        state.update_count += result.num_gradient_steps
        state.epoch += 1
        state.last_offline_batch_size = float(offline_batch_size)
        state.last_online_batch_size = float(online_batch_size)
        _sync_rlpd_trainer_state(trainer_state, state)
        callback_list.on_update_end(trainer_state, result)
        if _request_stop_for_limits(
            trainer_state,
            epoch=state.epoch,
            update_count=state.update_count,
            max_epochs=loop_config.max_epochs,
            max_updates=loop_config.max_updates,
        ):
            break
    return state


def train_rlpd(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="rlpd", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    gradient_updates_per_step = int(config.algo_kwargs.get("gradient_updates_per_step", 1))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    alpha = float(config.algo_kwargs.get("alpha", 0.2))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    offline_pretrain_updates = int(config.algo_kwargs.get("offline_pretrain_updates", 1000))
    offline_batch_ratio = _validate_offline_batch_ratio(config.algo_kwargs.get("offline_batch_ratio", 0.5))
    eval_interval = resolve_eval_interval(config)
    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    warmup_steps = int(config.algo_kwargs.get("warmup_steps", 0))
    learning_rate_schedule = config.algo_kwargs.get("learning_rate_schedule")
    effective_total_updates = _resolve_effective_total_rlpd_updates(
        config,
        offline_pretrain_updates=offline_pretrain_updates,
        gradient_updates_per_step=gradient_updates_per_step,
    )

    if buffer_capacity < 1:
        raise ValueError(f"buffer_capacity must be >= 1, got {buffer_capacity}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if learning_starts < 0:
        raise ValueError(f"learning_starts must be >= 0, got {learning_starts}")
    if train_frequency < 1:
        raise ValueError(f"train_frequency must be >= 1, got {train_frequency}")
    if gradient_updates_per_step < 1:
        raise ValueError(f"gradient_updates_per_step must be >= 1, got {gradient_updates_per_step}")
    if offline_pretrain_updates < 0:
        raise ValueError(f"offline_pretrain_updates must be >= 0, got {offline_pretrain_updates}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_dim, action_dim = _infer_spaces(envs)
        action_space = envs.single_action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for RLPD trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        offline_dataset = _build_offline_dataset(config, action_space=action_space)
        model = MLPSACModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = RLPD(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
        )
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=device,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        obs, _ = envs.reset(seed=config.seed)
        state = _restore_rlpd_training_state(
            checkpoint_state,
            batch_size=batch_size,
            offline_pretrain_updates=offline_pretrain_updates,
            learning_rate=learning_rate,
        )
        loop_config = _RLPDLoopConfig(
            learning_rate_schedule=learning_rate_schedule,
            effective_total_updates=effective_total_updates,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            device=device,
            alpha=alpha,
            offline_pretrain_updates=offline_pretrain_updates,
            offline_batch_ratio=offline_batch_ratio,
            learning_starts=learning_starts,
            train_frequency=train_frequency,
            gradient_updates_per_step=gradient_updates_per_step,
            max_epochs=max_epochs,
            max_updates=max_updates,
        )
        _sync_rlpd_trainer_state(trainer_state, state)
        _request_stop_for_limits(
            trainer_state,
            epoch=state.epoch,
            update_count=state.update_count,
            max_epochs=max_epochs,
            max_updates=max_updates,
        )
        callback_list.on_train_start(trainer_state)
        _emit_offline_dataset_event(
            callback_list,
            trainer_state,
            offline_dataset_size=len(offline_dataset),
            replay_buffer_size=len(replay_buffer),
        )

        state, metrics = _run_rlpd_pretraining(
            algorithm=algorithm,
            offline_dataset=offline_dataset,
            replay_buffer=replay_buffer,
            callback_list=callback_list,
            trainer_state=trainer_state,
            loop_config=loop_config,
            state=state,
        )

        while state.global_step < config.total_timesteps and not trainer_state.should_stop:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                normalized_actions = model.sample_actions(obs_tensor).actions
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            store_vector_transitions(
                replay_buffer,
                obs=obs,
                actions=normalized_actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                num_envs=config.num_envs,
            )

            obs = next_obs
            state.global_step += config.num_envs
            _sync_rlpd_trainer_state(trainer_state, state)
            _emit_online_collect_event(
                callback_list,
                trainer_state,
                global_step=state.global_step,
                dones=dones,
                offline_dataset_size=len(offline_dataset),
                replay_buffer=replay_buffer,
                obs=obs,
            )

            state = _run_rlpd_online_updates(
                algorithm=algorithm,
                offline_dataset=offline_dataset,
                replay_buffer=replay_buffer,
                callback_list=callback_list,
                trainer_state=trainer_state,
                loop_config=loop_config,
                global_step=state.global_step,
                state=state,
            )

            metrics = _build_rlpd_metrics(
                state,
                alpha=alpha,
                offline_dataset_size=len(offline_dataset),
                replay_buffer_size=len(replay_buffer),
                offline_pretrain_updates=offline_pretrain_updates,
                offline_batch_ratio=offline_batch_ratio,
            )
            metrics, should_stop = maybe_run_evaluation(
                should_run_eval=should_run_periodic_eval(
                    global_step=state.global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                evaluate=lambda: _evaluate_sac_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                ),
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=state.global_step,
            )
            if should_stop:
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=replay_buffer.state_dict(),
            trainer_state={
                "global_step": state.global_step,
                "epoch": state.epoch,
                "update_count": state.update_count,
                "pretrain_updates_done": state.pretrain_updates_done,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        if envs is not None:
            envs.close()
        session.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
