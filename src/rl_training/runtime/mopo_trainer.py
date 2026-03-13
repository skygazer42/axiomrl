from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.mopo import MOPO
from rl_training.data.offline_dataset import TransitionDataset
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    build_control_callbacks,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from rl_training.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.sac_trainer import _evaluate_sac_policy
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _validate_synthetic_batch_ratio(value: object) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"synthetic_batch_ratio must be between 0 and 1, got {resolved}")
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
    synthetic_batch_ratio: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    synthetic_batch_size = min(batch_size, max(0, int(round(batch_size * synthetic_batch_ratio))))
    real_batch_size = batch_size - synthetic_batch_size

    if len(replay_buffer) == 0:
        real_batch_size = batch_size
        synthetic_batch_size = 0

    batches: list[dict[str, torch.Tensor]] = []
    if real_batch_size > 0:
        batches.append(offline_dataset.sample(real_batch_size, device=device))
    if synthetic_batch_size > 0:
        batches.append(replay_buffer.sample(synthetic_batch_size))
    return _concatenate_transition_batches(batches), real_batch_size, synthetic_batch_size


def _resolve_effective_total_mopo_updates(config: TrainConfig, *, model_updates: int) -> int:
    candidates = [max(int(config.total_timesteps), 0) + max(int(model_updates), 0)]
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


def _apply_mopo_learning_rate_scale(algorithm: MOPO, *, scale: float) -> float:
    return apply_learning_rate_scale(
        [
            algorithm.model_optimizer,
            algorithm.policy_algorithm.actor_optimizer,
            algorithm.policy_algorithm.critic_optimizer,
        ],
        scale=scale,
    )


def _refresh_synthetic_buffer(
    *,
    algorithm: MOPO,
    offline_dataset: TransitionDataset,
    replay_buffer: ReplayBuffer,
    rollout_batch_size: int,
    rollout_horizon: int,
    device: torch.device,
) -> dict[str, float]:
    replay_buffer.reset()
    if rollout_batch_size < 1 or rollout_horizon < 1:
        return {
            "synthetic_buffer_size": 0.0,
            "synthetic_rollout_transitions": 0.0,
            "synthetic_reward_mean": 0.0,
            "synthetic_disagreement_mean": 0.0,
        }

    algorithm.set_eval_mode()
    current_obs = offline_dataset.sample(rollout_batch_size, device=device)["obs"]
    reward_means: list[float] = []
    disagreement_means: list[float] = []
    transitions_added = 0

    with torch.no_grad():
        for _ in range(rollout_horizon):
            actions = algorithm.policy_model.sample_actions(current_obs).actions
            synthetic = algorithm.sample_synthetic_transition(current_obs, actions)
            dones = torch.zeros(current_obs.shape[0], dtype=torch.float32, device=device)

            for index in range(int(current_obs.shape[0])):
                replay_buffer.add(
                    obs=current_obs[index],
                    actions=actions[index],
                    rewards=synthetic["rewards"][index],
                    next_obs=synthetic["next_obs"][index],
                    dones=dones[index],
                )
            reward_means.append(float(synthetic["rewards"].mean().detach().cpu().item()))
            disagreement_means.append(float(synthetic["disagreement"].mean().detach().cpu().item()))
            transitions_added += int(current_obs.shape[0])
            current_obs = synthetic["next_obs"]

    algorithm.set_train_mode()
    return {
        "synthetic_buffer_size": float(len(replay_buffer)),
        "synthetic_rollout_transitions": float(transitions_added),
        "synthetic_reward_mean": float(np.mean(reward_means)) if reward_means else 0.0,
        "synthetic_disagreement_mean": float(np.mean(disagreement_means)) if disagreement_means else 0.0,
    }


def train_mopo(
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
    trainer_state = TrainerState(algorithm="mopo", run_dir=run_context.run_dir)

    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", hidden_sizes))
    num_ensembles = int(config.algo_kwargs.get("num_ensembles", 5))
    model_batch_size = int(config.algo_kwargs.get("model_batch_size", batch_size))
    model_updates = int(config.algo_kwargs.get("model_updates", 1000))
    rollout_batch_size = int(config.algo_kwargs.get("rollout_batch_size", 1024))
    rollout_horizon = int(config.algo_kwargs.get("rollout_horizon", 3))
    rollout_refresh_interval = int(config.algo_kwargs.get("rollout_refresh_interval", 250))
    synthetic_buffer_capacity = int(config.algo_kwargs.get("synthetic_buffer_capacity", 100000))
    synthetic_batch_ratio = _validate_synthetic_batch_ratio(config.algo_kwargs.get("synthetic_batch_ratio", 0.5))
    policy_learning_rate = float(config.algo_kwargs.get("policy_learning_rate", 3e-4))
    model_learning_rate = float(config.algo_kwargs.get("model_learning_rate", 1e-3))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    alpha = float(config.algo_kwargs.get("alpha", 0.2))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    penalty_coef = float(config.algo_kwargs.get("penalty_coef", 1.0))
    eval_interval = resolve_eval_interval(config)
    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    warmup_steps = int(config.algo_kwargs.get("warmup_steps", 0))
    learning_rate_schedule = config.algo_kwargs.get("learning_rate_schedule")
    effective_total_updates = _resolve_effective_total_mopo_updates(config, model_updates=model_updates)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if model_batch_size < 1:
        raise ValueError(f"model_batch_size must be >= 1, got {model_batch_size}")
    if model_updates < 0:
        raise ValueError(f"model_updates must be >= 0, got {model_updates}")
    if rollout_batch_size < 1:
        raise ValueError(f"rollout_batch_size must be >= 1, got {rollout_batch_size}")
    if rollout_horizon < 1:
        raise ValueError(f"rollout_horizon must be >= 1, got {rollout_horizon}")
    if rollout_refresh_interval < 1:
        raise ValueError(f"rollout_refresh_interval must be >= 1, got {rollout_refresh_interval}")
    if synthetic_buffer_capacity < 1:
        raise ValueError(f"synthetic_buffer_capacity must be >= 1, got {synthetic_buffer_capacity}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_space, action_space = _infer_env_spaces(config)
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space for MOPO trainer: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for MOPO trainer: {type(action_space)!r}")

        offline_dataset = _build_offline_dataset(config, action_space=action_space)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])

        algorithm = MOPO(
            policy_model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
            dynamics_model=MLPMOPOEnsembleModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=model_hidden_sizes,
                num_ensembles=num_ensembles,
            ).to(device),
            policy_learning_rate=policy_learning_rate,
            model_learning_rate=model_learning_rate,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
            penalty_coef=penalty_coef,
        )
        synthetic_buffer = ReplayBuffer(
            capacity=synthetic_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=device,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                synthetic_buffer.load_state_dict(checkpoint_state.buffer_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", global_step)) if checkpoint_state is not None else 0
        epoch = int(checkpoint_state.trainer_state.get("epoch", update_count)) if checkpoint_state is not None else 0
        model_updates_done = (
            int(checkpoint_state.trainer_state.get("model_updates_done", min(update_count, model_updates)))
            if checkpoint_state is not None
            else 0
        )
        latest_update_metrics: MetricDict = {}
        latest_refresh_metrics: MetricDict = {
            "synthetic_buffer_size": float(len(synthetic_buffer)),
            "synthetic_rollout_transitions": float(len(synthetic_buffer)),
            "synthetic_reward_mean": 0.0,
            "synthetic_disagreement_mean": 0.0,
        }
        last_real_batch_size = float(batch_size)
        last_synthetic_batch_size = 0.0
        last_learning_rate = 0.0
        last_lr_scale = 1.0
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
                    "synthetic_buffer_size": float(len(synthetic_buffer)),
                },
                last_obs=None,
            ),
        )

        while model_updates_done < model_updates and not trainer_state.should_stop:
            last_lr_scale = resolve_schedule_value(
                learning_rate_schedule,
                step=update_count,
                total_steps=effective_total_updates,
                warmup_steps=warmup_steps,
            )
            last_learning_rate = _apply_mopo_learning_rate_scale(algorithm, scale=last_lr_scale)
            result = algorithm.update_model(offline_dataset.sample(model_batch_size, device=device), global_step=global_step)
            model_updates_done += result.num_gradient_steps
            update_count += result.num_gradient_steps
            epoch += 1
            latest_update_metrics = result.metrics
            trainer_state.epoch = epoch
            trainer_state.update_count = update_count
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                **latest_refresh_metrics,
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "offline_dataset_size": float(len(offline_dataset)),
                "synthetic_buffer_size": float(len(synthetic_buffer)),
                "buffer_size": float(len(synthetic_buffer)),
                "model_updates": float(model_updates),
                "model_updates_done": float(model_updates_done),
                "synthetic_batch_ratio": float(synthetic_batch_ratio),
                "real_batch_size": last_real_batch_size,
                "synthetic_batch_size": last_synthetic_batch_size,
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
            if synthetic_batch_ratio > 0.0 and (len(synthetic_buffer) == 0 or global_step % rollout_refresh_interval == 0):
                latest_refresh_metrics = _refresh_synthetic_buffer(
                    algorithm=algorithm,
                    offline_dataset=offline_dataset,
                    replay_buffer=synthetic_buffer,
                    rollout_batch_size=rollout_batch_size,
                    rollout_horizon=rollout_horizon,
                    device=device,
                )
                callback_list.on_collect_end(
                    trainer_state,
                    CollectResult(
                        num_env_steps=0,
                        num_episodes=0,
                        metrics=latest_refresh_metrics,
                        last_obs=None,
                    ),
                )

            last_lr_scale = resolve_schedule_value(
                learning_rate_schedule,
                step=update_count,
                total_steps=effective_total_updates,
                warmup_steps=warmup_steps,
            )
            last_learning_rate = _apply_mopo_learning_rate_scale(algorithm, scale=last_lr_scale)
            batch, real_batch_size, synthetic_batch_size = _sample_mixed_batch(
                offline_dataset=offline_dataset,
                replay_buffer=synthetic_buffer,
                batch_size=batch_size,
                synthetic_batch_ratio=synthetic_batch_ratio,
                device=device,
            )
            result = algorithm.update(batch, global_step=global_step)
            global_step += 1
            update_count += result.num_gradient_steps
            epoch += 1
            latest_update_metrics = result.metrics
            last_real_batch_size = float(real_batch_size)
            last_synthetic_batch_size = float(synthetic_batch_size)
            trainer_state.global_step = global_step
            trainer_state.epoch = epoch
            trainer_state.update_count = update_count
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                **latest_refresh_metrics,
                "alpha": float(alpha),
                "penalty_coef": float(penalty_coef),
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "offline_dataset_size": float(len(offline_dataset)),
                "synthetic_buffer_size": float(len(synthetic_buffer)),
                "buffer_size": float(len(synthetic_buffer)),
                "model_updates": float(model_updates),
                "model_updates_done": float(model_updates_done),
                "synthetic_batch_ratio": float(synthetic_batch_ratio),
                "real_batch_size": last_real_batch_size,
                "synthetic_batch_size": last_synthetic_batch_size,
                "lr_scale": float(last_lr_scale),
                "learning_rate": float(last_learning_rate),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_sac_policy(
                    algorithm.policy_model,
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

            stop_reason = stop_reason_for_training_limits(
                epoch=epoch,
                update_count=update_count,
                max_epochs=max_epochs,
                max_updates=max_updates,
            )
            if stop_reason is not None:
                trainer_state.request_stop(stop_reason)
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=synthetic_buffer.state_dict(),
            trainer_state={
                "global_step": global_step,
                "epoch": epoch,
                "update_count": update_count,
                "model_updates_done": model_updates_done,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        run_artifacts.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
