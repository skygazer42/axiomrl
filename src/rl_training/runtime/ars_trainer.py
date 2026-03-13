from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.ars import ARS
from rl_training.envs.factory import build_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_ars import MLPARSModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks, resolve_eval_interval, should_run_evaluation
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.td3_trainer import _action_bounds, _scale_actions
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(env: gym.Env) -> tuple[int, int]:
    obs_space = env.observation_space
    action_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for ARS trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for ARS trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.shape[0])


def _evaluate_ars_policy(
    model: MLPARSModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = build_env(config, 0, evaluation=True)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for ARS evaluation: {type(action_space)!r}")

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
                    normalized_action = model.actor(obs_tensor).squeeze(0)
                    env_action = _scale_actions(normalized_action, low=low, high=high)
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


def _run_rollout(
    env: gym.Env,
    model: MLPARSModel,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[float, int, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    episode_return = 0.0
    steps = 0

    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            normalized_action = model.actor(obs_tensor).squeeze(0)
            env_action = _scale_actions(normalized_action, low=low, high=high)
        obs, reward, done, truncated, _ = env.step(env_action.cpu().numpy())
        episode_return += float(reward)
        steps += 1

    return episode_return, steps, np.asarray(obs, dtype=np.float32)


def train_ars(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if config.num_envs != 1:
        raise ValueError(f"ars trainer currently supports num_envs=1 only, got {config.num_envs}")

    device = resolve_device(config.device)
    run_artifacts = create_training_run(config, run_suffix=run_suffix)
    run_context = run_artifacts.run_context
    logger = run_artifacts.logger
    callback_list = CallbackList(merge_callbacks(build_control_callbacks(config), callbacks))
    trainer_state = TrainerState(algorithm="ars", run_dir=run_context.run_dir)

    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    step_size = float(config.algo_kwargs.get("step_size", 0.02))
    noise_std = float(config.algo_kwargs.get("noise_std", 0.03))
    num_directions = int(config.algo_kwargs.get("num_directions", 8))
    num_top_directions = int(config.algo_kwargs.get("num_top_directions", max(1, num_directions // 2)))
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_env = build_env(config, 0, evaluation=False)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_dim, action_dim = _infer_spaces(train_env)
        action_space = train_env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for ARS trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        model = MLPARSModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        algorithm = ARS(
            model=model,
            step_size=step_size,
            noise_std=noise_std,
            num_top_directions=num_top_directions,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_index = int(checkpoint_state.trainer_state.get("update_index", 0)) if checkpoint_state is not None else 0
        rollout_index = int(checkpoint_state.trainer_state.get("rollout_index", 0)) if checkpoint_state is not None else 0
        trainer_state.global_step = global_step
        trainer_state.update_count = update_index
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            base_parameters = model.flat_parameters().detach().clone()
            perturbations = algorithm.sample_perturbations(num_directions)
            positive_returns: list[float] = []
            negative_returns: list[float] = []
            collected_steps = 0
            last_obs: np.ndarray | None = None

            for direction in perturbations:
                rollout_seed = config.seed + rollout_index
                model.set_flat_parameters(base_parameters + noise_std * direction)
                positive_return, positive_steps, last_obs = _run_rollout(
                    train_env,
                    model,
                    low=low,
                    high=high,
                    device=device,
                    seed=rollout_seed,
                )
                model.set_flat_parameters(base_parameters - noise_std * direction)
                negative_return, negative_steps, last_obs = _run_rollout(
                    train_env,
                    model,
                    low=low,
                    high=high,
                    device=device,
                    seed=rollout_seed,
                )
                positive_returns.append(positive_return)
                negative_returns.append(negative_return)
                collected_steps += positive_steps + negative_steps
                global_step += positive_steps + negative_steps
                rollout_index += 1
                trainer_state.global_step = global_step

            model.set_flat_parameters(base_parameters)
            callback_list.on_collect_end(
                trainer_state,
                CollectResult(
                    num_env_steps=collected_steps,
                    num_episodes=2 * num_directions,
                    metrics={"global_step": float(global_step)},
                    last_obs=last_obs,
                ),
            )

            result = algorithm.update(
                {
                    "perturbations": perturbations,
                    "positive_returns": torch.tensor(positive_returns, dtype=torch.float32, device=device),
                    "negative_returns": torch.tensor(negative_returns, dtype=torch.float32, device=device),
                },
                global_step=global_step,
            )
            update_index += result.num_gradient_steps
            trainer_state.update_count = update_index
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **result.metrics,
                "global_step": float(global_step),
                "update": float(update_index),
                "gradient_steps": float(update_index),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                eval_metrics = _evaluate_ars_policy(
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

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
            trainer_state={
                "global_step": global_step,
                "update_index": update_index,
                "rollout_index": rollout_index,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        train_env.close()
        run_artifacts.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
