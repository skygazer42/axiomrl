from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from axiomrl.algorithms.ddpg import DDPG
from axiomrl.data.replay_buffer import ReplayBuffer
from axiomrl.envs.factory import make_vector_env
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.mlp_ddpg import MLPDDPGModel
from axiomrl.runtime.callbacks import Callback, CallbackList
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import resolve_eval_interval, should_run_evaluation
from axiomrl.runtime.evaluation_support import evaluate_continuous_episodes
from axiomrl.runtime.resume_state import (
    capture_global_random_state,
    capture_vector_env_resume_state,
    restore_global_random_state,
    restore_vector_env_resume_state,
)
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.td3_trainer import _action_bounds, _apply_exploration_noise, _infer_spaces, _scale_actions
from axiomrl.runtime.trainer import TrainerState, TrainResult
from axiomrl.runtime.types import MetricDict


def _evaluate_ddpg_policy(
    model: MLPDDPGModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    eval_env = make_vector_env(replace(config, num_envs=1, execution_backend="local_sync"))
    action_space = eval_env.single_action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for DDPG evaluation: {type(action_space)!r}")
    low, high = _action_bounds(action_space, device=device)
    try:

        def action_fn(obs_tensor: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                normalized_action = model.actor(obs_tensor).squeeze(0)
                env_action = _scale_actions(normalized_action, low=low, high=high)
            return env_action.cpu().numpy()

        return evaluate_continuous_episodes(
            config,
            device=device,
            num_episodes=num_episodes,
            action_fn=action_fn,
        )
    finally:
        eval_env.close()


def _store_vector_transitions(
    replay_buffer: ReplayBuffer,
    *,
    obs: np.ndarray,
    normalized_actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        replay_buffer.add(
            obs=obs[env_index],
            actions=normalized_actions[env_index],
            rewards=float(rewards[env_index]),
            next_obs=next_obs[env_index],
            dones=float(dones[env_index]),
        )


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    num_envs: int,
    dones: np.ndarray,
    replay_buffer: ReplayBuffer,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=num_envs,
            num_episodes=int(np.sum(dones)),
            metrics={"global_step": float(global_step), "buffer_size": float(len(replay_buffer))},
            last_obs=obs,
        ),
    )


def _maybe_update_algorithm(
    algorithm: DDPG,
    replay_buffer: ReplayBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    global_step: int,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int]:
    if len(replay_buffer) < max(batch_size, learning_starts):
        return latest_update_metrics, update_count
    if global_step % train_frequency != 0:
        return latest_update_metrics, update_count

    result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps


def _restore_training_state(
    *,
    algorithm: DDPG,
    replay_buffer: ReplayBuffer,
    envs: gym.vector.VectorEnv,
    checkpoint_state: CheckpointState | None,
) -> tuple[np.ndarray | None, int, int]:
    if checkpoint_state is None:
        return None, 0, 0
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    if checkpoint_state.buffer_state is not None:
        replay_buffer.load_state_dict(checkpoint_state.buffer_state)
    restored_obs = None
    resume_context = checkpoint_state.trainer_state.get("resume_context")
    if isinstance(resume_context, dict):
        env_resume_state = resume_context.get("env_state")
        if isinstance(env_resume_state, dict):
            restored_obs = restore_vector_env_resume_state(envs, env_resume_state)
        random_state = resume_context.get("random_state")
        if isinstance(random_state, dict):
            restore_global_random_state(random_state)
    return (
        restored_obs,
        int(checkpoint_state.trainer_state.get("global_step", 0)),
        int(checkpoint_state.trainer_state.get("update_count", 0)),
    )


def _build_metrics(
    latest_update_metrics: MetricDict,
    *,
    global_step: int,
    replay_buffer: ReplayBuffer,
    update_count: int,
) -> MetricDict:
    return {
        **latest_update_metrics,
        "global_step": float(global_step),
        "buffer_size": float(len(replay_buffer)),
        "gradient_steps": float(update_count),
    }


def _maybe_run_evaluation(
    algorithm: DDPG,
    model: MLPDDPGModel,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    config: TrainConfig,
    logger: object,
    *,
    device: torch.device,
    global_step: int,
    eval_interval: int,
    metrics: MetricDict,
) -> tuple[MetricDict, bool]:
    if not should_run_evaluation(
        global_step=global_step,
        total_timesteps=config.total_timesteps,
        eval_interval=eval_interval,
    ):
        return metrics, False

    algorithm.set_eval_mode()
    eval_metrics = _evaluate_ddpg_policy(
        model,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    algorithm.set_train_mode()

    merged_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(merged_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, merged_metrics)
    return merged_metrics, trainer_state.should_stop


def train_ddpg(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="ddpg", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    exploration_noise = float(config.algo_kwargs.get("exploration_noise", 0.0))
    eval_interval = resolve_eval_interval(config)

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
            raise TypeError(f"unsupported action space for DDPG trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        model = MLPDDPGModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = DDPG(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
        )
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=device,
        )
        obs, _ = envs.reset(seed=config.seed)
        restored_obs, global_step, update_count = _restore_training_state(
            algorithm=algorithm,
            replay_buffer=replay_buffer,
            envs=envs,
            checkpoint_state=checkpoint_state,
        )
        if restored_obs is not None:
            obs = restored_obs
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                normalized_actions = model.actor(obs_tensor)
                normalized_actions = _apply_exploration_noise(normalized_actions, std=exploration_noise)
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _store_vector_transitions(
                replay_buffer,
                obs=obs,
                normalized_actions=normalized_actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                num_envs=config.num_envs,
            )

            obs = next_obs
            global_step += config.num_envs
            trainer_state.global_step = global_step
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                num_envs=config.num_envs,
                dones=dones,
                replay_buffer=replay_buffer,
                obs=obs,
            )

            latest_update_metrics, update_count = _maybe_update_algorithm(
                algorithm,
                replay_buffer,
                callback_list,
                trainer_state,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_frequency=train_frequency,
                global_step=global_step,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )
            trainer_state.update_count = update_count

            metrics = _build_metrics(
                latest_update_metrics,
                global_step=global_step,
                replay_buffer=replay_buffer,
                update_count=update_count,
            )
            metrics, should_stop = _maybe_run_evaluation(
                algorithm,
                model,
                callback_list,
                trainer_state,
                config,
                logger,
                device=device,
                global_step=global_step,
                eval_interval=eval_interval,
                metrics=metrics,
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
                "update_count": update_count,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
                "resume_context": {
                    "env_state": capture_vector_env_resume_state(envs),
                    "random_state": capture_global_random_state(),
                },
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
