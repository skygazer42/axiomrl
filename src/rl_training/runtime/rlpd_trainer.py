from __future__ import annotations

import math
from collections.abc import Sequence
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
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    build_control_callbacks,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_periodic_eval,
    stop_reason_for_training_limits,
)
from rl_training.runtime.iql_trainer import _build_offline_dataset
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.sac_trainer import _action_bounds, _evaluate_sac_policy, _infer_spaces, _scale_actions
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


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


def train_rlpd(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    device = resolve_device(config.device)
    run_artifacts = create_training_run(config, run_suffix=run_suffix)
    run_context = run_artifacts.run_context
    logger = run_artifacts.logger
    callback_list = CallbackList(merge_callbacks(build_control_callbacks(config), callbacks))
    trainer_state = TrainerState(algorithm="rlpd", run_dir=run_context.run_dir)

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

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
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
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = (
            int(checkpoint_state.trainer_state.get("update_count", 0))
            if checkpoint_state is not None
            else 0
        )
        epoch = int(checkpoint_state.trainer_state.get("epoch", update_count)) if checkpoint_state is not None else 0
        pretrain_updates_done = (
            int(checkpoint_state.trainer_state.get("pretrain_updates_done", min(update_count, offline_pretrain_updates)))
            if checkpoint_state is not None
            else 0
        )
        latest_update_metrics: MetricDict = {}
        last_offline_batch_size = float(batch_size if pretrain_updates_done > 0 else 0.0)
        last_online_batch_size = 0.0
        last_lr_scale = 1.0
        last_learning_rate = learning_rate
        trainer_state.global_step = global_step
        trainer_state.epoch = epoch
        trainer_state.update_count = update_count
        initial_stop_reason = stop_reason_for_training_limits(
            epoch=epoch,
            update_count=update_count,
            max_epochs=max_epochs,
            max_updates=max_updates,
        )
        if initial_stop_reason is not None:
            trainer_state.request_stop(initial_stop_reason)
        callback_list.on_train_start(trainer_state)
        callback_list.on_collect_end(
            trainer_state,
            CollectResult(
                num_env_steps=len(offline_dataset),
                num_episodes=0,
                metrics={
                    "offline_dataset_size": float(len(offline_dataset)),
                    "online_buffer_size": float(len(replay_buffer)),
                },
                last_obs=None,
            ),
        )

        while pretrain_updates_done < offline_pretrain_updates and not trainer_state.should_stop:
            last_lr_scale = resolve_schedule_value(
                learning_rate_schedule,
                step=update_count,
                total_steps=effective_total_updates,
                warmup_steps=warmup_steps,
            )
            last_learning_rate = apply_learning_rate_scale(algorithm, scale=last_lr_scale)
            result = algorithm.update(offline_dataset.sample(batch_size, device=device), global_step=global_step)
            pretrain_updates_done += 1
            update_count += result.num_gradient_steps
            epoch += 1
            latest_update_metrics = result.metrics
            last_offline_batch_size = float(batch_size)
            last_online_batch_size = 0.0
            trainer_state.global_step = global_step
            trainer_state.epoch = epoch
            trainer_state.update_count = update_count
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "alpha": alpha,
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "offline_dataset_size": float(len(offline_dataset)),
                "online_buffer_size": float(len(replay_buffer)),
                "buffer_size": float(len(replay_buffer)),
                "offline_pretrain_updates": float(offline_pretrain_updates),
                "pretrain_updates_done": float(pretrain_updates_done),
                "offline_batch_ratio": float(offline_batch_ratio),
                "offline_batch_size": last_offline_batch_size,
                "online_batch_size": last_online_batch_size,
                "lr_scale": float(last_lr_scale),
                "learning_rate": float(last_learning_rate),
            }
            stop_reason = stop_reason_for_training_limits(
                epoch=epoch,
                update_count=update_count,
                max_epochs=max_epochs,
                max_updates=max_updates,
            )
            if stop_reason is not None:
                trainer_state.request_stop(stop_reason)
                break

        while global_step < config.total_timesteps and not trainer_state.should_stop:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                normalized_actions = model.sample_actions(obs_tensor).actions
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            for env_index in range(config.num_envs):
                replay_buffer.add(
                    obs=obs[env_index],
                    actions=normalized_actions[env_index],
                    rewards=float(rewards[env_index]),
                    next_obs=next_obs[env_index],
                    dones=float(dones[env_index]),
                )

            obs = next_obs
            global_step += config.num_envs
            trainer_state.global_step = global_step
            callback_list.on_collect_end(
                trainer_state,
                CollectResult(
                    num_env_steps=config.num_envs,
                    num_episodes=int(np.sum(dones)),
                    metrics={
                        "global_step": float(global_step),
                        "offline_dataset_size": float(len(offline_dataset)),
                        "online_buffer_size": float(len(replay_buffer)),
                    },
                    last_obs=obs,
                ),
            )

            if len(replay_buffer) >= learning_starts and global_step % train_frequency == 0:
                for _ in range(gradient_updates_per_step):
                    last_lr_scale = resolve_schedule_value(
                        learning_rate_schedule,
                        step=update_count,
                        total_steps=effective_total_updates,
                        warmup_steps=warmup_steps,
                    )
                    last_learning_rate = apply_learning_rate_scale(algorithm, scale=last_lr_scale)
                    batch, offline_batch_size, online_batch_size = _sample_mixed_batch(
                        offline_dataset=offline_dataset,
                        replay_buffer=replay_buffer,
                        batch_size=batch_size,
                        offline_batch_ratio=offline_batch_ratio,
                        device=device,
                    )
                    result = algorithm.update(batch, global_step=global_step)
                    latest_update_metrics = result.metrics
                    update_count += result.num_gradient_steps
                    epoch += 1
                    last_offline_batch_size = float(offline_batch_size)
                    last_online_batch_size = float(online_batch_size)
                    trainer_state.epoch = epoch
                    trainer_state.update_count = update_count
                    callback_list.on_update_end(trainer_state, result)

                    stop_reason = stop_reason_for_training_limits(
                        epoch=epoch,
                        update_count=update_count,
                        max_epochs=max_epochs,
                        max_updates=max_updates,
                    )
                    if stop_reason is not None:
                        trainer_state.request_stop(stop_reason)
                        break

            metrics = {
                **latest_update_metrics,
                "alpha": alpha,
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "offline_dataset_size": float(len(offline_dataset)),
                "online_buffer_size": float(len(replay_buffer)),
                "buffer_size": float(len(replay_buffer)),
                "offline_pretrain_updates": float(offline_pretrain_updates),
                "pretrain_updates_done": float(pretrain_updates_done),
                "offline_batch_ratio": float(offline_batch_ratio),
                "offline_batch_size": last_offline_batch_size,
                "online_batch_size": last_online_batch_size,
                "lr_scale": float(last_lr_scale),
                "learning_rate": float(last_learning_rate),
            }
            if should_run_periodic_eval(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_sac_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                )
                algorithm.set_train_mode()
                metrics = {**metrics, **eval_metrics}
                logger.log_metrics(metrics, step=global_step)
                callback_list.on_eval_end(trainer_state, metrics)
                if trainer_state.should_stop:
                    break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=replay_buffer.state_dict(),
            trainer_state={
                "global_step": global_step,
                "epoch": epoch,
                "update_count": update_count,
                "pretrain_updates_done": pretrain_updates_done,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        envs.close()
        run_artifacts.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
