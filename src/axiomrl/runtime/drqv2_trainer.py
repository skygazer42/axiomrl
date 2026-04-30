from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.drqv2 import DrQv2
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn.drqv2 import CNNDrQv2Model
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import resolve_eval_interval, should_run_evaluation
from rl_training.runtime.evaluation_support import evaluate_continuous_episodes
from rl_training.runtime.off_policy_trainer_utils import (
    capture_replay_resume_context,
    restore_replay_training_state,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.td3_trainer import _action_bounds, _scale_actions
from rl_training.runtime.trainer import TrainerState, TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for DrQ-v2 trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for DrQ-v2 trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 3:
        raise ValueError(f"expected channel-first image observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.shape[0])


def _evaluate_drqv2_policy(
    model: CNNDrQv2Model,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    class _ActionFn:
        def __init__(self) -> None:
            self.low: torch.Tensor | None = None
            self.high: torch.Tensor | None = None

        def bind_env(self, env: gym.Env) -> None:
            action_space = env.action_space
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError(f"unsupported action space for DrQ-v2 evaluation: {type(action_space)!r}")
            self.low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
            self.high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)

        def __call__(self, obs_tensor: torch.Tensor) -> np.ndarray:
            if self.low is None or self.high is None:
                raise RuntimeError("action bounds must be bound before evaluation")
            with torch.no_grad():
                normalized_action = model.actor(obs_tensor).squeeze(0)
                env_action = _scale_actions(normalized_action, low=self.low, high=self.high)
            return env_action.cpu().numpy()

    return evaluate_continuous_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=_ActionFn(),
    )


def _store_replay_transitions(
    replay_buffer: ReplayBuffer,
    *,
    obs: np.ndarray,
    actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        replay_buffer.add(
            obs=obs[env_index],
            actions=actions[env_index],
            rewards=float(rewards[env_index]),
            next_obs=next_obs[env_index],
            dones=float(dones[env_index]),
        )


def _maybe_update_algorithm(
    *,
    algorithm: DrQv2,
    replay_buffer: ReplayBuffer,
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

    result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps


def _maybe_run_evaluation(
    *,
    algorithm: DrQv2,
    model: CNNDrQv2Model,
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
    eval_metrics = _evaluate_drqv2_policy(
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


def train_drqv2(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="drqv2", run_suffix=run_suffix, callbacks=callbacks)
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
    tau = float(config.algo_kwargs.get("tau", 0.01))
    policy_delay = int(config.algo_kwargs.get("policy_delay", 2))
    augmentation_pad = int(config.algo_kwargs.get("augmentation_pad", 4))
    exploration_noise = float(config.algo_kwargs.get("exploration_noise", 0.1))
    exploration_noise_clip = float(config.algo_kwargs.get("exploration_noise_clip", 0.3))
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
            raise TypeError(f"unsupported observation space for DrQ-v2 trainer: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space for DrQ-v2 trainer: {type(action_space)!r}")

        low, high = _action_bounds(action_space, device=device)
        obs_dtype = torch.uint8 if np.issubdtype(obs_space.dtype, np.integer) else torch.float32

        model = CNNDrQv2Model(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=features_dim,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
        ).to(device)
        algorithm = DrQv2(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
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
                normalized_actions = model.sample_actions(
                    obs_tensor,
                    std=exploration_noise,
                    clip=exploration_noise_clip,
                ).actions
                env_actions = _scale_actions(normalized_actions, low=low, high=high)

            next_obs, rewards, terminated, truncated, _ = envs.step(env_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _store_replay_transitions(
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
                batch_size=batch_size,
                learning_starts=learning_starts,
                global_step=global_step,
                train_frequency=train_frequency,
                callback_list=callback_list,
                trainer_state=trainer_state,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )
            trainer_state.update_count = update_count

            metrics = {
                **latest_update_metrics,
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
