from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.sac import SAC
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState, save_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for SAC trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for SAC trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.shape[0])


def _action_bounds(space: gym.spaces.Box, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.as_tensor(space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(space.high, dtype=torch.float32, device=device)
    return low, high


def _scale_actions(normalized_actions: torch.Tensor, *, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scaled = low + 0.5 * (normalized_actions + 1.0) * (high - low)
    return torch.max(torch.min(scaled, high), low)


def _evaluate_sac_policy(
    model: MLPSACModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = gym.make(config.env_id, **config.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for SAC evaluation: {type(action_space)!r}")

    low, high = _action_bounds(action_space, device=device)
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


def train_sac(
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
    trainer_state = TrainerState(algorithm="sac", run_dir=run_context.run_dir)

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    alpha = float(config.algo_kwargs.get("alpha", 0.2))
    tau = float(config.algo_kwargs.get("tau", 0.005))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_dim, action_dim = _infer_spaces(envs)
        action_space = envs.single_action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for SAC trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        model = MLPSACModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = SAC(
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
        update_count = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
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
                    metrics={"global_step": float(global_step), "buffer_size": float(len(replay_buffer))},
                    last_obs=obs,
                ),
            )

            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
                latest_update_metrics = result.metrics
                update_count += result.num_gradient_steps
                callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "alpha": alpha,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
            }

        eval_metrics = _evaluate_sac_policy(
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
            buffer_state=replay_buffer.state_dict(),
            trainer_state={"global_step": global_step},
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
