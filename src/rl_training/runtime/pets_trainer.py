from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.pets import PETS
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import build_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks, resolve_eval_interval, should_run_evaluation
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(env: gym.Env) -> tuple[int, int]:
    obs_space = env.observation_space
    action_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for PETS trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for PETS trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.shape[0])


def _planner_kwargs(config: TrainConfig) -> dict[str, int]:
    horizon = int(config.algo_kwargs.get("planning_horizon", 5))
    num_candidates = int(config.algo_kwargs.get("planning_candidates", 256))
    num_iterations = int(config.algo_kwargs.get("planning_iterations", 4))
    num_topk = int(config.algo_kwargs.get("planning_topk", 32))
    num_particles = int(config.algo_kwargs.get("planning_particles", 8))

    if horizon < 1:
        raise ValueError(f"planning_horizon must be >= 1, got {horizon}")
    if num_candidates < 1:
        raise ValueError(f"planning_candidates must be >= 1, got {num_candidates}")
    if num_iterations < 1:
        raise ValueError(f"planning_iterations must be >= 1, got {num_iterations}")
    if num_topk < 1 or num_topk > num_candidates:
        raise ValueError(f"planning_topk must be in [1, {num_candidates}], got {num_topk}")
    if num_particles < 1:
        raise ValueError(f"planning_particles must be >= 1, got {num_particles}")

    return {
        "horizon": horizon,
        "num_candidates": num_candidates,
        "num_iterations": num_iterations,
        "num_topk": num_topk,
        "num_particles": num_particles,
    }


def _validate_pets_hyperparameters(
    *,
    num_envs: int,
    buffer_capacity: int,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    num_ensembles: int,
    model_updates_per_step: int,
    initial_random_steps: int,
) -> None:
    if num_envs != 1:
        raise ValueError(f"pets trainer currently supports num_envs=1 only, got {num_envs}")
    if buffer_capacity < 1:
        raise ValueError(f"buffer_capacity must be >= 1, got {buffer_capacity}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if learning_starts < 0:
        raise ValueError(f"learning_starts must be >= 0, got {learning_starts}")
    if train_frequency < 1:
        raise ValueError(f"train_frequency must be >= 1, got {train_frequency}")
    if num_ensembles < 2:
        raise ValueError(f"num_ensembles must be >= 2, got {num_ensembles}")
    if model_updates_per_step < 1:
        raise ValueError(f"model_updates_per_step must be >= 1, got {model_updates_per_step}")
    if initial_random_steps < 0:
        raise ValueError(f"initial_random_steps must be >= 0, got {initial_random_steps}")


def _sample_random_action(action_space: gym.spaces.Box) -> np.ndarray:
    return np.asarray(action_space.sample(), dtype=np.float32)


def _evaluate_pets_policy(
    algorithm: PETS,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    del device
    env = build_env(config, 0, evaluation=True)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for PETS evaluation: {type(action_space)!r}")

    planner_config = _planner_kwargs(config)
    returns: list[float] = []

    try:
        algorithm.set_eval_mode()
        for episode_index in range(num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_index)
            done = False
            truncated = False
            episode_return = 0.0

            while not (done or truncated):
                action = algorithm.plan_action(
                    obs,
                    action_low=action_space.low,
                    action_high=action_space.high,
                    deterministic=True,
                    **planner_config,
                )
                obs, reward, done, truncated, _ = env.step(action)
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_episodes": float(len(returns)),
    }


def _select_pets_action(
    algorithm: PETS,
    action_space: gym.spaces.Box,
    *,
    obs: np.ndarray,
    global_step: int,
    initial_random_steps: int,
    replay_size: int,
    batch_size: int,
    learning_starts: int,
    planner_config: dict[str, int],
) -> np.ndarray:
    if global_step < initial_random_steps or replay_size < max(batch_size, learning_starts):
        return _sample_random_action(action_space)

    return algorithm.plan_action(
        obs,
        action_low=action_space.low,
        action_high=action_space.high,
        deterministic=False,
        **planner_config,
    )


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    done: bool,
    replay_buffer: ReplayBuffer,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=1,
            num_episodes=int(done),
            metrics={
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
            },
            last_obs=obs,
        ),
    )


def _maybe_update_pets_model(
    algorithm: PETS,
    replay_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    model_updates_per_step: int,
    global_step: int,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int]:
    if len(replay_buffer) < max(batch_size, learning_starts) or global_step % train_frequency != 0:
        return latest_update_metrics, update_count

    algorithm.set_train_mode()
    current_metrics = latest_update_metrics
    current_update_count = update_count
    for _ in range(model_updates_per_step):
        result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
        current_metrics = result.metrics
        current_update_count += result.num_gradient_steps
        trainer_state.update_count = current_update_count
        callback_list.on_update_end(trainer_state, result)
    return current_metrics, current_update_count


def _build_pets_metrics(
    latest_update_metrics: MetricDict,
    *,
    global_step: int,
    replay_buffer: ReplayBuffer,
    update_count: int,
    done: bool,
    episode_return: float,
    episode_length: int,
) -> MetricDict:
    metrics: MetricDict = {
        **latest_update_metrics,
        "global_step": float(global_step),
        "buffer_size": float(len(replay_buffer)),
        "gradient_steps": float(update_count),
    }
    if done:
        metrics["episode_return"] = float(episode_return)
        metrics["episode_length"] = float(episode_length)
    return metrics


def _maybe_run_pets_evaluation(
    *,
    should_run_eval: bool,
    algorithm: PETS,
    config: TrainConfig,
    device: torch.device,
    logger: object,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    metrics: MetricDict,
    global_step: int,
) -> tuple[MetricDict, bool]:
    if not should_run_eval:
        return metrics, False

    algorithm.set_eval_mode()
    eval_metrics = _evaluate_pets_policy(
        algorithm,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    algorithm.set_train_mode()
    evaluated_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(evaluated_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, evaluated_metrics)
    return evaluated_metrics, trainer_state.should_stop


def _reset_episode_if_done(
    train_env: gym.Env,
    config: TrainConfig,
    *,
    done: bool,
    episode_index: int,
) -> tuple[np.ndarray, int, float, int] | None:
    if not done:
        return None

    next_episode_index = episode_index + 1
    obs, _ = train_env.reset(seed=config.seed + next_episode_index)
    return np.asarray(obs, dtype=np.float32), next_episode_index, 0.0, 0


def train_pets(
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
    trainer_state = TrainerState(algorithm="pets", run_dir=run_context.run_dir)

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", (256, 256)))
    model_learning_rate = float(config.algo_kwargs.get("model_learning_rate", 1e-3))
    num_ensembles = int(config.algo_kwargs.get("num_ensembles", 5))
    model_updates_per_step = int(config.algo_kwargs.get("model_updates_per_step", 1))
    initial_random_steps = int(config.algo_kwargs.get("initial_random_steps", learning_starts))
    eval_interval = resolve_eval_interval(config)
    planner_config = _planner_kwargs(config)

    _validate_pets_hyperparameters(
        num_envs=config.num_envs,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_frequency=train_frequency,
        num_ensembles=num_ensembles,
        model_updates_per_step=model_updates_per_step,
        initial_random_steps=initial_random_steps,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_env = build_env(config, 0, evaluation=False)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_dim, action_dim = _infer_spaces(train_env)
        action_space = train_env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for PETS trainer: {type(action_space)!r}")

        algorithm = PETS(
            dynamics_model=MLPMOPOEnsembleModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=model_hidden_sizes,
                num_ensembles=num_ensembles,
            ).to(device),
            learning_rate=model_learning_rate,
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

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        episode_index = int(checkpoint_state.trainer_state.get("episode_index", 0)) if checkpoint_state is not None else 0
        obs, _ = train_env.reset(seed=config.seed + episode_index)
        episode_return = 0.0
        episode_length = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            action = _select_pets_action(
                algorithm,
                action_space,
                obs=obs,
                global_step=global_step,
                initial_random_steps=initial_random_steps,
                replay_size=len(replay_buffer),
                batch_size=batch_size,
                learning_starts=learning_starts,
                planner_config=planner_config,
            )

            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = bool(terminated or truncated)
            replay_buffer.add(
                obs=obs,
                actions=action,
                rewards=float(reward),
                next_obs=next_obs,
                dones=float(done),
            )

            obs = np.asarray(next_obs, dtype=np.float32)
            episode_return += float(reward)
            episode_length += 1
            global_step += 1
            trainer_state.global_step = global_step
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                done=done,
                replay_buffer=replay_buffer,
                obs=obs,
            )

            latest_update_metrics, update_count = _maybe_update_pets_model(
                algorithm,
                replay_buffer,
                callback_list,
                trainer_state,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_frequency=train_frequency,
                model_updates_per_step=model_updates_per_step,
                global_step=global_step,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )

            metrics = _build_pets_metrics(
                latest_update_metrics,
                global_step=global_step,
                replay_buffer=replay_buffer,
                update_count=update_count,
                done=done,
                episode_return=episode_return,
                episode_length=episode_length,
            )

            metrics, should_stop = _maybe_run_pets_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                config=config,
                device=device,
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=global_step,
            )
            if should_stop:
                break

            reset_state = _reset_episode_if_done(
                train_env,
                config,
                done=done,
                episode_index=episode_index,
            )
            if reset_state is not None:
                obs, episode_index, episode_return, episode_length = reset_state

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=replay_buffer.state_dict(),
            trainer_state={
                "global_step": global_step,
                "update_count": update_count,
                "episode_index": episode_index,
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
