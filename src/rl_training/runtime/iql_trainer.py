from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import torch

from rl_training.data.dataset_loaders import load_transition_dataset
from rl_training.data.offline_mixers import mix_transition_datasets
from rl_training.algorithms.iql import IQL
from rl_training.data.offline_dataset import TransitionDataset
from rl_training.envs.factory import build_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_iql import MLPIQLModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    build_control_callbacks,
    resolve_effective_total_updates,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_env_spaces(config: TrainConfig) -> tuple[gym.spaces.Box, gym.spaces.Box]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space for IQL trainer: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for IQL trainer: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return obs_space, action_space
    finally:
        env.close()


def _action_bounds(space: gym.spaces.Box) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32)


def _normalize_actions(actions: np.ndarray, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    scale = np.where(np.abs(high - low) < 1e-8, 1.0, high - low)
    normalized = 2.0 * (actions - low) / scale - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _scale_actions(normalized_actions: torch.Tensor, *, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scaled = low + 0.5 * (normalized_actions + 1.0) * (high - low)
    return torch.max(torch.min(scaled, high), low)


def _build_random_transition_dataset(config: TrainConfig) -> TransitionDataset:
    dataset_size = int(config.algo_kwargs.get("dataset_size", 10000))
    dataset_seed = int(config.algo_kwargs.get("dataset_seed", config.seed))
    if dataset_size < 1:
        raise ValueError(f"dataset_size must be >= 1, got {dataset_size}")

    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space for IQL dataset generation: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for IQL dataset generation: {type(action_space)!r}")

        low, high = _action_bounds(action_space)
        action_space.seed(dataset_seed)
        obs, _ = env.reset(seed=dataset_seed)

        obs_buffer = np.zeros((dataset_size, *obs_space.shape), dtype=np.float32)
        actions_buffer = np.zeros((dataset_size, *action_space.shape), dtype=np.float32)
        rewards_buffer = np.zeros((dataset_size,), dtype=np.float32)
        next_obs_buffer = np.zeros((dataset_size, *obs_space.shape), dtype=np.float32)
        dones_buffer = np.zeros((dataset_size,), dtype=np.float32)
        next_actions_buffer = np.zeros((dataset_size, *action_space.shape), dtype=np.float32)

        current_action = np.asarray(action_space.sample(), dtype=np.float32)

        for index in range(dataset_size):
            next_obs, reward, terminated, truncated, _ = env.step(current_action)
            done = bool(terminated or truncated)
            next_action = (
                np.zeros_like(current_action, dtype=np.float32)
                if done
                else np.asarray(action_space.sample(), dtype=np.float32)
            )

            obs_buffer[index] = np.asarray(obs, dtype=np.float32)
            actions_buffer[index] = _normalize_actions(current_action, low=low, high=high)
            rewards_buffer[index] = float(reward)
            next_obs_buffer[index] = np.asarray(next_obs, dtype=np.float32)
            dones_buffer[index] = float(done)
            next_actions_buffer[index] = _normalize_actions(next_action, low=low, high=high)

            if done:
                obs, _ = env.reset()
                current_action = np.asarray(action_space.sample(), dtype=np.float32)
            else:
                obs = next_obs
                current_action = next_action

        return TransitionDataset(
            obs=obs_buffer,
            actions=actions_buffer,
            rewards=rewards_buffer,
            next_obs=next_obs_buffer,
            dones=dones_buffer,
            next_actions=next_actions_buffer,
        )
    finally:
        env.close()


def _build_file_transition_dataset(config: TrainConfig) -> TransitionDataset:
    return load_transition_dataset(
        str(config.algo_kwargs.get("dataset_kind", "npz")),
        dataset_path=config.algo_kwargs.get("dataset_path"),
        dataset_id=config.algo_kwargs.get("dataset_id"),
        download=bool(config.algo_kwargs.get("dataset_download", False)),
    )


def _build_mixed_offline_dataset(config: TrainConfig, *, action_space: gym.spaces.Box) -> TransitionDataset:
    mix_payload = config.algo_kwargs.get("dataset_mix")
    if not isinstance(mix_payload, Sequence) or isinstance(mix_payload, (str, bytes)):
        raise TypeError(f"expected algo_kwargs['dataset_mix'] to be a sequence of mappings, got {type(mix_payload)!r}")
    if not mix_payload:
        raise ValueError("algo_kwargs['dataset_mix'] must not be empty")

    component_datasets: list[TransitionDataset] = []
    weights: list[float] = []
    for index, descriptor in enumerate(mix_payload):
        if not isinstance(descriptor, Mapping):
            raise TypeError(
                f"expected dataset_mix[{index}] to be a mapping, got {type(descriptor)!r}"
            )
        entry_kwargs = dict(config.algo_kwargs)
        entry_kwargs.pop("dataset_mix", None)
        entry_kwargs.pop("dataset_mix_size", None)
        entry_kwargs.pop("dataset_mix_seed", None)

        for key, value in descriptor.items():
            if key == "weight":
                continue
            if key == "kind":
                entry_kwargs["dataset_kind"] = value
                continue
            entry_kwargs[str(key)] = value

        entry_kwargs.setdefault("dataset_seed", config.seed + index + 1)
        entry_config = cast(TrainConfig, replace(config, algo_kwargs=entry_kwargs))
        component_datasets.append(_build_offline_dataset(entry_config, action_space=action_space))
        weights.append(float(descriptor.get("weight", 1.0)))

    return mix_transition_datasets(
        component_datasets,
        weights=weights,
        total_size=int(config.algo_kwargs.get("dataset_mix_size", sum(len(dataset) for dataset in component_datasets))),
        seed=int(config.algo_kwargs.get("dataset_mix_seed", config.seed)),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _process_offline_dataset(
    dataset: TransitionDataset,
    *,
    config: TrainConfig,
    action_space: gym.spaces.Box,
    dataset_kind: str,
) -> TransitionDataset:
    normalize_dataset_actions = bool(config.algo_kwargs.get("normalize_dataset_actions", dataset_kind != "random"))
    if normalize_dataset_actions:
        normalized_actions = _normalize_actions(
            dataset.actions.cpu().numpy().astype(np.float32),
            low=np.asarray(action_space.low, dtype=np.float32),
            high=np.asarray(action_space.high, dtype=np.float32),
        )
        dataset = TransitionDataset(
            obs=dataset.obs,
            actions=normalized_actions,
            rewards=dataset.rewards,
            next_obs=dataset.next_obs,
            dones=dataset.dones,
            next_actions=(
                _normalize_actions(
                    dataset.next_actions.cpu().numpy().astype(np.float32),
                    low=np.asarray(action_space.low, dtype=np.float32),
                    high=np.asarray(action_space.high, dtype=np.float32),
                )
                if dataset.next_actions is not None
                else None
            ),
            returns_to_go=dataset.returns_to_go,
        )

    return dataset.with_reward_transform(
        scale=float(config.algo_kwargs.get("reward_scale", 1.0)),
        shift=float(config.algo_kwargs.get("reward_shift", 0.0)),
        clip_min=_optional_float(config.algo_kwargs.get("reward_clip_min")),
        clip_max=_optional_float(config.algo_kwargs.get("reward_clip_max")),
    )


def _build_offline_dataset(config: TrainConfig, *, action_space: gym.spaces.Box | None = None) -> TransitionDataset:
    if config.algo_kwargs.get("dataset_mix") not in (None, False):
        resolved_action_space = action_space
        if resolved_action_space is None:
            _, resolved_action_space = _infer_env_spaces(config)
        return _build_mixed_offline_dataset(config, action_space=resolved_action_space)

    dataset_kind = str(config.algo_kwargs.get("dataset_kind", "random"))
    if dataset_kind == "random":
        dataset = _build_random_transition_dataset(config)
    elif dataset_kind in {"npz", "pt", "pth", "torch", "minari"}:
        dataset = _build_file_transition_dataset(config)
    else:
        raise ValueError(f"unsupported IQL dataset_kind: {dataset_kind!r}")

    resolved_action_space = action_space
    if resolved_action_space is None:
        _, resolved_action_space = _infer_env_spaces(config)
    return _process_offline_dataset(dataset, config=config, action_space=resolved_action_space, dataset_kind=dataset_kind)


def _evaluate_iql_policy(
    model: MLPIQLModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = build_env(config, 0, evaluation=True)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for IQL evaluation: {type(action_space)!r}")

    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    returns: list[float] = []

    try:
        for episode_index in range(num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_index)
            done = False
            truncated = False
            episode_return = 0.0

            while not (done or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    normalized_action = model.sample_actions(obs_tensor, deterministic=True).actions
                    env_action = _scale_actions(normalized_action, low=low, high=high).squeeze(0)
                obs, reward, done, truncated, _ = env.step(env_action.cpu().numpy())
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_episodes": float(len(returns)),
    }


def train_iql(
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
    trainer_state = TrainerState(algorithm="iql", run_dir=run_context.run_dir)

    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    expectile = float(config.algo_kwargs.get("expectile", 0.7))
    beta = float(config.algo_kwargs.get("beta", 3.0))
    max_advantage_weight = float(config.algo_kwargs.get("max_advantage_weight", 100.0))
    eval_interval = resolve_eval_interval(config)
    effective_total_updates = resolve_effective_total_updates(config)
    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    warmup_steps = int(config.algo_kwargs.get("warmup_steps", 0))
    learning_rate_schedule = config.algo_kwargs.get("learning_rate_schedule")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_space, action_space = _infer_env_spaces(config)
        dataset = _build_offline_dataset(config, action_space=action_space)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])

        model = MLPIQLModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = IQL(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            expectile=expectile,
            beta=beta,
            max_advantage_weight=max_advantage_weight,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        epoch = int(checkpoint_state.trainer_state.get("epoch", global_step)) if checkpoint_state is not None else 0
        update_count = (
            int(checkpoint_state.trainer_state.get("update_count", global_step))
            if checkpoint_state is not None
            else 0
        )
        latest_update_metrics: MetricDict = {}
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
                num_env_steps=len(dataset),
                num_episodes=0,
                metrics={"dataset_size": float(len(dataset))},
                last_obs=None,
            ),
        )

        while global_step < config.total_timesteps and not trainer_state.should_stop:
            lr_scale = resolve_schedule_value(
                learning_rate_schedule,
                step=update_count,
                total_steps=effective_total_updates,
                warmup_steps=warmup_steps,
            )
            current_learning_rate = apply_learning_rate_scale(algorithm, scale=lr_scale)
            result = algorithm.update(dataset.sample(batch_size, device=device), global_step=global_step)
            global_step += 1
            epoch += 1
            update_count += result.num_gradient_steps
            latest_update_metrics = result.metrics
            trainer_state.global_step = global_step
            trainer_state.epoch = epoch
            trainer_state.update_count = update_count
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "dataset_size": float(len(dataset)),
                "lr_scale": float(lr_scale),
                "learning_rate": float(current_learning_rate),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                eval_metrics = _evaluate_iql_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                )
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
            buffer_state=None,
            trainer_state={
                "global_step": global_step,
                "epoch": epoch,
                "update_count": update_count,
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
