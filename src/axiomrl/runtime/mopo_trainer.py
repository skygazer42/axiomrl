from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from axiomrl.algorithms.mopo import MOPO
from axiomrl.data.offline_dataset import TransitionDataset
from axiomrl.data.replay_buffer import ReplayBuffer
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.mlp_mopo import MLPMOPOEnsembleModel
from axiomrl.models.mlp_sac import MLPSACModel
from axiomrl.runtime.callbacks import Callback, CallbackList
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import (
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from axiomrl.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces
from axiomrl.runtime.off_policy_trainer_utils import maybe_run_evaluation
from axiomrl.runtime.resume_state import capture_global_random_state, restore_global_random_state
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.sac_trainer import _evaluate_sac_policy
from axiomrl.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainerState, TrainResult
from axiomrl.runtime.types import MetricDict


@dataclass
class _MOPOTrainingState:
    global_step: int
    update_count: int
    epoch: int
    model_updates_done: int
    latest_update_metrics: MetricDict
    latest_refresh_metrics: MetricDict
    last_real_batch_size: float
    last_synthetic_batch_size: float
    last_learning_rate: float
    last_lr_scale: float


@dataclass(frozen=True)
class _MOPOLoopConfig:
    learning_rate_schedule: object
    effective_total_updates: int
    warmup_steps: int
    model_batch_size: int
    batch_size: int
    device: torch.device
    alpha: float
    penalty_coef: float
    model_updates: int
    synthetic_batch_ratio: float
    max_epochs: int | None
    max_updates: int | None
    rollout_batch_size: int
    rollout_horizon: int
    rollout_refresh_interval: int


def _validate_synthetic_batch_ratio(value: object) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"synthetic_batch_ratio must be between 0 and 1, got {resolved}")
    return resolved


def _validate_mopo_hyperparameters(
    *,
    batch_size: int,
    model_batch_size: int,
    model_updates: int,
    rollout_batch_size: int,
    rollout_horizon: int,
    rollout_refresh_interval: int,
    synthetic_buffer_capacity: int,
) -> None:
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


def _restore_mopo_training_state(
    checkpoint_state: CheckpointState | None,
    *,
    batch_size: int,
    model_updates: int,
) -> _MOPOTrainingState:
    if checkpoint_state is None:
        return _MOPOTrainingState(
            global_step=0,
            update_count=0,
            epoch=0,
            model_updates_done=0,
            latest_update_metrics={},
            latest_refresh_metrics={
                "synthetic_buffer_size": 0.0,
                "synthetic_rollout_transitions": 0.0,
                "synthetic_reward_mean": 0.0,
                "synthetic_disagreement_mean": 0.0,
            },
            last_real_batch_size=float(batch_size),
            last_synthetic_batch_size=0.0,
            last_learning_rate=0.0,
            last_lr_scale=1.0,
        )

    update_count = int(
        checkpoint_state.trainer_state.get("update_count", int(checkpoint_state.trainer_state.get("global_step", 0)))
    )
    return _MOPOTrainingState(
        global_step=int(checkpoint_state.trainer_state.get("global_step", 0)),
        update_count=update_count,
        epoch=int(checkpoint_state.trainer_state.get("epoch", update_count)),
        model_updates_done=int(
            checkpoint_state.trainer_state.get("model_updates_done", min(update_count, model_updates))
        ),
        latest_update_metrics={},
        latest_refresh_metrics={
            "synthetic_buffer_size": 0.0,
            "synthetic_rollout_transitions": 0.0,
            "synthetic_reward_mean": 0.0,
            "synthetic_disagreement_mean": 0.0,
        },
        last_real_batch_size=float(batch_size),
        last_synthetic_batch_size=0.0,
        last_learning_rate=0.0,
        last_lr_scale=1.0,
    )


def _sync_mopo_trainer_state(trainer_state: TrainerState, state: _MOPOTrainingState) -> None:
    trainer_state.global_step = state.global_step
    trainer_state.epoch = state.epoch
    trainer_state.update_count = state.update_count


def _build_mopo_metrics(
    state: _MOPOTrainingState,
    *,
    alpha: float,
    penalty_coef: float,
    offline_dataset_size: int,
    synthetic_buffer_size: int,
    model_updates: int,
    synthetic_batch_ratio: float,
) -> MetricDict:
    return {
        **state.latest_update_metrics,
        **state.latest_refresh_metrics,
        "alpha": float(alpha),
        "penalty_coef": float(penalty_coef),
        "global_step": float(state.global_step),
        "epoch": float(state.epoch),
        "update_count": float(state.update_count),
        "gradient_steps": float(state.update_count),
        "offline_dataset_size": float(offline_dataset_size),
        "synthetic_buffer_size": float(synthetic_buffer_size),
        "buffer_size": float(synthetic_buffer_size),
        "model_updates": float(model_updates),
        "model_updates_done": float(state.model_updates_done),
        "synthetic_batch_ratio": float(synthetic_batch_ratio),
        "real_batch_size": state.last_real_batch_size,
        "synthetic_batch_size": state.last_synthetic_batch_size,
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
    synthetic_buffer_size: int,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=offline_dataset_size,
            num_episodes=0,
            metrics={
                "offline_dataset_size": float(offline_dataset_size),
                "synthetic_buffer_size": float(synthetic_buffer_size),
            },
            last_obs=None,
        ),
    )


def _emit_refresh_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    refresh_metrics: MetricDict,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=0,
            num_episodes=0,
            metrics=refresh_metrics,
            last_obs=None,
        ),
    )


def _run_mopo_model_pretraining(
    *,
    algorithm: MOPO,
    offline_dataset: TransitionDataset,
    synthetic_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    loop_config: _MOPOLoopConfig,
    state: _MOPOTrainingState,
) -> tuple[_MOPOTrainingState, MetricDict]:
    metrics = _build_mopo_metrics(
        state,
        alpha=loop_config.alpha,
        penalty_coef=loop_config.penalty_coef,
        offline_dataset_size=len(offline_dataset),
        synthetic_buffer_size=len(synthetic_buffer),
        model_updates=loop_config.model_updates,
        synthetic_batch_ratio=loop_config.synthetic_batch_ratio,
    )
    while state.model_updates_done < loop_config.model_updates and not trainer_state.should_stop:
        state.last_lr_scale = resolve_schedule_value(
            loop_config.learning_rate_schedule,
            step=state.update_count,
            total_steps=loop_config.effective_total_updates,
            warmup_steps=loop_config.warmup_steps,
        )
        state.last_learning_rate = _apply_mopo_learning_rate_scale(algorithm, scale=state.last_lr_scale)
        result = algorithm.update_model(
            offline_dataset.sample(loop_config.model_batch_size, device=loop_config.device),
            global_step=state.global_step,
        )
        state.model_updates_done += result.num_gradient_steps
        state.update_count += result.num_gradient_steps
        state.epoch += 1
        state.latest_update_metrics = result.metrics
        _sync_mopo_trainer_state(trainer_state, state)
        callback_list.on_update_end(trainer_state, result)

        metrics = _build_mopo_metrics(
            state,
            alpha=loop_config.alpha,
            penalty_coef=loop_config.penalty_coef,
            offline_dataset_size=len(offline_dataset),
            synthetic_buffer_size=len(synthetic_buffer),
            model_updates=loop_config.model_updates,
            synthetic_batch_ratio=loop_config.synthetic_batch_ratio,
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


def _maybe_refresh_synthetic_rollouts(
    *,
    algorithm: MOPO,
    offline_dataset: TransitionDataset,
    synthetic_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    loop_config: _MOPOLoopConfig,
    state: _MOPOTrainingState,
) -> _MOPOTrainingState:
    should_refresh = loop_config.synthetic_batch_ratio > 0.0 and (
        len(synthetic_buffer) == 0 or state.global_step % loop_config.rollout_refresh_interval == 0
    )
    if not should_refresh:
        return state

    state.latest_refresh_metrics = _refresh_synthetic_buffer(
        algorithm=algorithm,
        offline_dataset=offline_dataset,
        replay_buffer=synthetic_buffer,
        rollout_batch_size=loop_config.rollout_batch_size,
        rollout_horizon=loop_config.rollout_horizon,
        device=loop_config.device,
    )
    _emit_refresh_event(
        callback_list,
        trainer_state,
        refresh_metrics=state.latest_refresh_metrics,
    )
    return state


def _run_mopo_policy_step(
    *,
    algorithm: MOPO,
    offline_dataset: TransitionDataset,
    synthetic_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    loop_config: _MOPOLoopConfig,
    state: _MOPOTrainingState,
) -> _MOPOTrainingState:
    state.last_lr_scale = resolve_schedule_value(
        loop_config.learning_rate_schedule,
        step=state.update_count,
        total_steps=loop_config.effective_total_updates,
        warmup_steps=loop_config.warmup_steps,
    )
    state.last_learning_rate = _apply_mopo_learning_rate_scale(algorithm, scale=state.last_lr_scale)
    batch, real_batch_size, synthetic_batch_size = _sample_mixed_batch(
        offline_dataset=offline_dataset,
        replay_buffer=synthetic_buffer,
        batch_size=loop_config.batch_size,
        synthetic_batch_ratio=loop_config.synthetic_batch_ratio,
        device=loop_config.device,
    )
    result = algorithm.update(batch, global_step=state.global_step)
    state.global_step += 1
    state.update_count += result.num_gradient_steps
    state.epoch += 1
    state.latest_update_metrics = result.metrics
    state.last_real_batch_size = float(real_batch_size)
    state.last_synthetic_batch_size = float(synthetic_batch_size)
    _sync_mopo_trainer_state(trainer_state, state)
    callback_list.on_update_end(trainer_state, result)
    return state


def train_mopo(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="mopo", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

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

    _validate_mopo_hyperparameters(
        batch_size=batch_size,
        model_batch_size=model_batch_size,
        model_updates=model_updates,
        rollout_batch_size=rollout_batch_size,
        rollout_horizon=rollout_horizon,
        rollout_refresh_interval=rollout_refresh_interval,
        synthetic_buffer_capacity=synthetic_buffer_capacity,
    )

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

        state = _restore_mopo_training_state(
            checkpoint_state,
            batch_size=batch_size,
            model_updates=model_updates,
        )
        if checkpoint_state is not None:
            resume_context = checkpoint_state.trainer_state.get("resume_context")
            if isinstance(resume_context, dict):
                random_state = resume_context.get("random_state")
                if isinstance(random_state, dict):
                    restore_global_random_state(random_state)
        loop_config = _MOPOLoopConfig(
            learning_rate_schedule=learning_rate_schedule,
            effective_total_updates=effective_total_updates,
            warmup_steps=warmup_steps,
            model_batch_size=model_batch_size,
            batch_size=batch_size,
            device=device,
            alpha=alpha,
            penalty_coef=penalty_coef,
            model_updates=model_updates,
            synthetic_batch_ratio=synthetic_batch_ratio,
            max_epochs=max_epochs,
            max_updates=max_updates,
            rollout_batch_size=rollout_batch_size,
            rollout_horizon=rollout_horizon,
            rollout_refresh_interval=rollout_refresh_interval,
        )
        state.latest_refresh_metrics["synthetic_buffer_size"] = float(len(synthetic_buffer))
        state.latest_refresh_metrics["synthetic_rollout_transitions"] = float(len(synthetic_buffer))
        _sync_mopo_trainer_state(trainer_state, state)
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
            synthetic_buffer_size=len(synthetic_buffer),
        )

        state, metrics = _run_mopo_model_pretraining(
            algorithm=algorithm,
            offline_dataset=offline_dataset,
            synthetic_buffer=synthetic_buffer,
            callback_list=callback_list,
            trainer_state=trainer_state,
            loop_config=loop_config,
            state=state,
        )

        while state.global_step < config.total_timesteps and not trainer_state.should_stop:
            state = _maybe_refresh_synthetic_rollouts(
                algorithm=algorithm,
                offline_dataset=offline_dataset,
                synthetic_buffer=synthetic_buffer,
                callback_list=callback_list,
                trainer_state=trainer_state,
                loop_config=loop_config,
                state=state,
            )
            state = _run_mopo_policy_step(
                algorithm=algorithm,
                offline_dataset=offline_dataset,
                synthetic_buffer=synthetic_buffer,
                callback_list=callback_list,
                trainer_state=trainer_state,
                loop_config=loop_config,
                state=state,
            )

            metrics = _build_mopo_metrics(
                state,
                alpha=alpha,
                penalty_coef=penalty_coef,
                offline_dataset_size=len(offline_dataset),
                synthetic_buffer_size=len(synthetic_buffer),
                model_updates=model_updates,
                synthetic_batch_ratio=synthetic_batch_ratio,
            )
            metrics, should_stop = maybe_run_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=state.global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                evaluate=lambda: _evaluate_sac_policy(
                    algorithm.policy_model,
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

            if _request_stop_for_limits(
                trainer_state,
                epoch=state.epoch,
                update_count=state.update_count,
                max_epochs=max_epochs,
                max_updates=max_updates,
            ):
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=synthetic_buffer.state_dict(),
            trainer_state={
                "global_step": state.global_step,
                "epoch": state.epoch,
                "update_count": state.update_count,
                "model_updates_done": state.model_updates_done,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
                "resume_context": {
                    "random_state": capture_global_random_state(),
                },
            },
            metrics=metrics,
        )
    finally:
        session.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
