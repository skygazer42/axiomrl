from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.iql import IQL
from rl_training.data.offline_dataset import TransitionDataset
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_iql import MLPIQLModel
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_env_spaces(config: TrainConfig) -> tuple[gym.spaces.Box, gym.spaces.Box]:
    env = gym.make(config.env_id, **config.env_kwargs)
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
    normalized = 2.0 * (actions - low) / (high - low) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _scale_actions(normalized_actions: torch.Tensor, *, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scaled = low + 0.5 * (normalized_actions + 1.0) * (high - low)
    return torch.max(torch.min(scaled, high), low)


def _build_random_transition_dataset(config: TrainConfig) -> TransitionDataset:
    dataset_size = int(config.algo_kwargs.get("dataset_size", 10000))
    dataset_seed = int(config.algo_kwargs.get("dataset_seed", config.seed))
    if dataset_size < 1:
        raise ValueError(f"dataset_size must be >= 1, got {dataset_size}")

    env = gym.make(config.env_id, **config.env_kwargs)
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

        for index in range(dataset_size):
            env_action = np.asarray(action_space.sample(), dtype=np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = bool(terminated or truncated)

            obs_buffer[index] = np.asarray(obs, dtype=np.float32)
            actions_buffer[index] = _normalize_actions(env_action, low=low, high=high)
            rewards_buffer[index] = float(reward)
            next_obs_buffer[index] = np.asarray(next_obs, dtype=np.float32)
            dones_buffer[index] = float(done)

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return TransitionDataset(
            obs=obs_buffer,
            actions=actions_buffer,
            rewards=rewards_buffer,
            next_obs=next_obs_buffer,
            dones=dones_buffer,
        )
    finally:
        env.close()


def _build_offline_dataset(config: TrainConfig) -> TransitionDataset:
    dataset_kind = str(config.algo_kwargs.get("dataset_kind", "random"))
    if dataset_kind != "random":
        raise ValueError(f"unsupported IQL dataset_kind: {dataset_kind!r}")
    return _build_random_transition_dataset(config)


def _evaluate_iql_policy(
    model: MLPIQLModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = gym.make(config.env_id, **config.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
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
    callback_list = CallbackList(callbacks)
    trainer_state = TrainerState(algorithm="iql", run_dir=run_context.run_dir)

    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    expectile = float(config.algo_kwargs.get("expectile", 0.7))
    beta = float(config.algo_kwargs.get("beta", 3.0))
    max_advantage_weight = float(config.algo_kwargs.get("max_advantage_weight", 100.0))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_space, action_space = _infer_env_spaces(config)
        dataset = _build_offline_dataset(config)
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
        update_count = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
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

        while global_step < config.total_timesteps:
            result = algorithm.update(dataset.sample(batch_size, device=device), global_step=global_step)
            global_step += 1
            update_count += result.num_gradient_steps
            latest_update_metrics = result.metrics
            trainer_state.global_step = global_step
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "global_step": float(global_step),
                "gradient_steps": float(update_count),
                "dataset_size": float(len(dataset)),
            }

        eval_metrics = _evaluate_iql_policy(
            model,
            config,
            device=device,
            num_episodes=config.eval_episodes,
        )
        metrics = {**metrics, **eval_metrics}
        logger.log_metrics(metrics, step=global_step)
        callback_list.on_eval_end(trainer_state, eval_metrics)

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
            trainer_state={"global_step": global_step},
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
