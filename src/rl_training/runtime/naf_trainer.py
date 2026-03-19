from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.naf import NAF
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_naf import MLPNAFModel
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.controls import resolve_eval_interval, should_run_evaluation
from rl_training.runtime.evaluation_support import evaluate_continuous_episodes
from rl_training.runtime.off_policy_trainer_utils import (
    build_replay_metrics,
    emit_collect_event,
    maybe_run_evaluation,
    maybe_update_algorithm,
    store_vector_transitions,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.td3_trainer import _action_bounds, _apply_exploration_noise, _infer_spaces, _scale_actions
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _evaluate_naf_policy(
    model: MLPNAFModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    eval_env = make_vector_env(replace(config, num_envs=1, execution_backend="local_sync"))
    action_space = eval_env.single_action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for NAF evaluation: {type(action_space)!r}")

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


def train_naf(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="naf", run_suffix=run_suffix, callbacks=callbacks)
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
            raise TypeError(f"unsupported action space for NAF trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        model = MLPNAFModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = NAF(
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
                normalized_actions = model.actor(obs_tensor)
                normalized_actions = _apply_exploration_noise(normalized_actions, std=exploration_noise)
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

            metrics = build_replay_metrics(
                latest_update_metrics,
                global_step=global_step,
                replay_buffer=replay_buffer,
                update_count=update_count,
            )
            metrics, should_stop = maybe_run_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                evaluate=lambda: _evaluate_naf_policy(
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
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
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
