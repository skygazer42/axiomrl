from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.her import HER
from rl_training.data.her_replay_buffer import HERReplayBuffer
from rl_training.envs.factory import build_env, make_vector_env
from rl_training.envs.goals import (
    GoalSpaceSpec,
    flatten_goal_observation,
    infer_goal_space_spec,
    is_goal_observation_space,
)
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_ddpg import MLPDDPGModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks, resolve_eval_interval, should_run_evaluation
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.td3_trainer import _action_bounds, _apply_exploration_noise, _scale_actions
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_her_spaces(config: TrainConfig) -> tuple[GoalSpaceSpec, int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not is_goal_observation_space(obs_space):
            raise TypeError(f"unsupported observation space for HER trainer: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for HER trainer: {type(action_space)!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return infer_goal_space_spec(obs_space), int(action_space.shape[0])
    finally:
        env.close()


def _slice_goal_observation(observation: Mapping[str, np.ndarray], env_index: int) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value[env_index], dtype=np.float32)
        for key, value in observation.items()
    }


def _evaluate_her_policy(
    model: MLPDDPGModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = build_env(config, 0, evaluation=True)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for HER evaluation: {type(action_space)!r}")

    low, high = _action_bounds(action_space, device=device)
    returns: list[float] = []
    successes: list[float] = []

    try:
        for episode_index in range(num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_index)
            done = False
            truncated = False
            episode_return = 0.0

            while not (done or truncated):
                obs_tensor = torch.as_tensor(flatten_goal_observation(obs), dtype=torch.float32, device=device)
                with torch.no_grad():
                    normalized_action = model.actor(obs_tensor).squeeze(0)
                    env_action = _scale_actions(normalized_action, low=low, high=high)
                obs, reward, done, truncated, _ = env.step(env_action.cpu().numpy())
                episode_return += float(reward)

            returns.append(episode_return)
            successes.append(float(done and not truncated))
    finally:
        env.close()

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_success_rate": float(np.mean(successes)) if successes else 0.0,
        "eval_episodes": float(len(returns)),
    }


def _restore_training_state(
    *,
    algorithm: HER,
    replay_buffer: HERReplayBuffer,
    checkpoint_state: CheckpointState | None,
) -> int:
    if checkpoint_state is None:
        return 0
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    if checkpoint_state.buffer_state is not None:
        replay_buffer.load_state_dict(checkpoint_state.buffer_state)
    return int(checkpoint_state.trainer_state.get("global_step", 0))


def _as_goal_observation_mapping(obs: object, *, source: str) -> Mapping[str, np.ndarray]:
    if not isinstance(obs, Mapping):
        raise TypeError(f"expected goal-conditioned Mapping observations from {source}, got {type(obs)!r}")
    return obs


def _store_her_transitions(
    replay_buffer: HERReplayBuffer,
    *,
    obs: Mapping[str, np.ndarray],
    actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: Mapping[str, np.ndarray],
    terminated: np.ndarray,
    truncated: np.ndarray,
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        replay_buffer.add(
            env_index=env_index,
            obs=_slice_goal_observation(obs, env_index),
            actions=actions[env_index].detach().cpu().numpy(),
            rewards=float(rewards[env_index]),
            next_obs=_slice_goal_observation(next_obs, env_index),
            terminated=bool(terminated[env_index]),
            truncated=bool(truncated[env_index]),
        )


def _maybe_update_algorithm(
    *,
    algorithm: HER,
    replay_buffer: HERReplayBuffer,
    reward_env: gym.Env,
    batch_size: int,
    learning_starts: int,
    global_step: int,
    train_frequency: int,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int]:
    if len(replay_buffer) < max(batch_size, learning_starts):
        return latest_update_metrics, update_count
    if global_step % train_frequency != 0:
        return latest_update_metrics, update_count

    result = algorithm.update(replay_buffer.sample(batch_size, env=reward_env), global_step=global_step)
    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps


def _maybe_run_evaluation(
    *,
    algorithm: HER,
    model: MLPDDPGModel,
    config: TrainConfig,
    logger,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    metrics: MetricDict,
    global_step: int,
    eval_interval: int,
    device: torch.device,
) -> tuple[MetricDict, bool]:
    if not should_run_evaluation(
        global_step=global_step,
        total_timesteps=config.total_timesteps,
        eval_interval=eval_interval,
    ):
        return metrics, False

    algorithm.set_eval_mode()
    eval_metrics = _evaluate_her_policy(
        model,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    algorithm.set_train_mode()
    evaluated_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(evaluated_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, evaluated_metrics)
    return evaluated_metrics, trainer_state.should_stop


def train_her(
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
    trainer_state = TrainerState(algorithm="her", run_dir=run_context.run_dir)

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 50000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    her_ratio = float(config.algo_kwargs.get("her_ratio", 0.8))
    goal_selection_strategy = str(config.algo_kwargs.get("goal_selection_strategy", "future"))
    exploration_noise = float(config.algo_kwargs.get("exploration_noise", 0.0))
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    reward_env = build_env(config, 0)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        goal_spec, action_dim = _infer_her_spaces(config)
        action_space = envs.single_action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for HER trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        model = MLPDDPGModel(
            obs_dim=goal_spec.flat_observation_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = HER(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
        )
        replay_buffer = HERReplayBuffer(
            capacity=buffer_capacity,
            num_envs=config.num_envs,
            obs_shape=(goal_spec.observation_dim,),
            goal_shape=(goal_spec.goal_dim,),
            action_shape=(action_dim,),
            her_ratio=her_ratio,
            goal_selection_strategy=goal_selection_strategy,
            device=device,
        )

        initial_obs, _ = envs.reset(seed=config.seed)
        obs = _as_goal_observation_mapping(initial_obs, source="vector env")
        global_step = _restore_training_state(
            algorithm=algorithm,
            replay_buffer=replay_buffer,
            checkpoint_state=checkpoint_state,
        )
        update_count = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            obs_tensor = torch.as_tensor(flatten_goal_observation(obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                normalized_actions = model.actor(obs_tensor)
                normalized_actions = _apply_exploration_noise(normalized_actions, std=exploration_noise)
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            raw_next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            next_obs = _as_goal_observation_mapping(raw_next_obs, source="vector env")
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _store_her_transitions(
                replay_buffer,
                obs=obs,
                actions=normalized_actions,
                rewards=rewards,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                num_envs=config.num_envs,
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

            latest_update_metrics, update_count = _maybe_update_algorithm(
                algorithm=algorithm,
                replay_buffer=replay_buffer,
                reward_env=reward_env,
                batch_size=batch_size,
                learning_starts=learning_starts,
                global_step=global_step,
                train_frequency=train_frequency,
                callback_list=callback_list,
                trainer_state=trainer_state,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )

            metrics = {
                **latest_update_metrics,
                "her_ratio": her_ratio,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
            }
            metrics, should_stop = _maybe_run_evaluation(
                algorithm=algorithm,
                model=model,
                config=config,
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=global_step,
                eval_interval=eval_interval,
                device=device,
            )
            if should_stop:
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=replay_buffer.state_dict(),
            trainer_state={
                "global_step": global_step,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        envs.close()
        reward_env.close()
        run_artifacts.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
