from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.drq import DrQ
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn.drq import CNNDrQModel
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.controls import resolve_eval_interval, should_run_periodic_eval
from rl_training.runtime.evaluation_support import evaluate_continuous_episodes
from rl_training.runtime.off_policy_trainer_utils import (
    build_replay_metrics,
    capture_replay_resume_context,
    emit_collect_event,
    maybe_run_evaluation,
    maybe_update_algorithm,
    restore_replay_training_state,
    store_vector_transitions,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.td3_trainer import _action_bounds, _scale_actions
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for DrQ trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for DrQ trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 3:
        raise ValueError(f"expected channel-first image observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.shape[0])


def _evaluate_drq_policy(
    model: CNNDrQModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    eval_env = make_vector_env(replace(config, num_envs=1, execution_backend="local_sync"))
    action_space = eval_env.single_action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for DrQ evaluation: {type(action_space)!r}")

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


def train_drq(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="drq", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    features_dim = int(config.algo_kwargs.get("features_dim", 256))
    actor_hidden_sizes = tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256)))
    critic_hidden_sizes = tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    alpha = float(config.algo_kwargs.get("alpha", 0.1))
    tau = float(config.algo_kwargs.get("tau", 0.01))
    augmentation_pad = int(config.algo_kwargs.get("augmentation_pad", 4))
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_shape, action_dim = _infer_spaces(envs)
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space for DrQ trainer: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for DrQ trainer: {type(action_space)!r}")

        low, high = _action_bounds(action_space, device=device)
        obs_dtype = torch.uint8 if np.issubdtype(obs_space.dtype, np.integer) else torch.float32

        model = CNNDrQModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=features_dim,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
        ).to(device)
        algorithm = DrQ(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
            augmentation_pad=augmentation_pad,
        )
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            action_shape=(action_dim,),
            device=device,
            obs_dtype=obs_dtype,
        )
        obs, _ = envs.reset(seed=config.seed)
        restored_obs, global_step, update_count = restore_replay_training_state(
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
                normalized_actions = model.sample_actions(obs_tensor).actions
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            store_vector_transitions(
                replay_buffer,
                obs=obs,
                actions=normalized_actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                num_envs=config.num_envs,
            )

            obs = next_obs
            global_step += config.num_envs
            trainer_state.global_step = global_step
            emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                num_envs=config.num_envs,
                dones=dones,
                replay_buffer=replay_buffer,
                obs=obs,
            )

            latest_update_metrics, update_count = maybe_update_algorithm(
                algorithm=algorithm,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_frequency=train_frequency,
                global_step=global_step,
                callback_list=callback_list,
                trainer_state=trainer_state,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )
            trainer_state.update_count = update_count

            metrics = build_replay_metrics(
                latest_update_metrics,
                global_step=global_step,
                replay_buffer=replay_buffer,
                update_count=update_count,
                extra_metrics={"alpha": alpha},
            )
            metrics, should_stop = maybe_run_evaluation(
                should_run_eval=should_run_periodic_eval(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                evaluate=lambda: _evaluate_drq_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                ),
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=global_step,
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
                "resume_context": capture_replay_resume_context(envs),
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
