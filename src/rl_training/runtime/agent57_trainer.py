from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from rl_training.algorithms.agent57 import Agent57
from rl_training.data.n_step import NStepAccumulator
from rl_training.data.prioritized_recurrent_replay_buffer import PrioritizedRecurrentReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.rnd import RNDModel
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.controls import (
    resolve_eval_interval,
    resolve_exploration_epsilon,
    should_run_evaluation,
)
from rl_training.runtime.r2d2_trainer import (
    _append_recurrent_transitions,
    _beta_at_step,
    _build_q_network,
    _build_r2d2_metrics,
    _capture_metadata_buffers,
    _capture_rollout_state,
    _emit_collect_event,
    _evaluate_r2d2_policy,
    _infer_spaces,
    _maybe_run_r2d2_evaluation,
    _PrioritizedReplaySchedule,
    _restore_metadata_buffers,
    _restore_rollout_state,
)
from rl_training.runtime.resume_state import (
    capture_global_random_state,
    capture_vector_env_resume_state,
    restore_global_random_state,
    restore_vector_env_resume_state,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainerState, TrainResult
from rl_training.runtime.types import MetricDict


def _build_rnd_model(config: TrainConfig, *, obs_shape: tuple[int, ...]) -> RNDModel:
    return RNDModel(
        obs_shape=obs_shape,
        hidden_sizes=tuple(config.algo_kwargs.get("rnd_hidden_sizes", (256,))),
        embedding_dim=int(config.algo_kwargs.get("rnd_embedding_dim", 128)),
    )


def _maybe_update_agent57(
    *,
    algorithm: Agent57,
    replay_buffer: PrioritizedRecurrentReplayBuffer,
    batch_size: int,
    learning_starts: int,
    train_frequency: int,
    global_step: int,
    prioritized_replay_schedule: _PrioritizedReplaySchedule,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int, float]:
    beta = _beta_at_step(
        global_step,
        total_timesteps=prioritized_replay_schedule.total_timesteps,
        beta_start=prioritized_replay_schedule.beta_start,
        beta_end=prioritized_replay_schedule.beta_end,
        beta_fraction=prioritized_replay_schedule.beta_fraction,
    )
    if (
        replay_buffer.num_transitions < learning_starts
        or len(replay_buffer) < batch_size
        or global_step % train_frequency != 0
    ):
        return latest_update_metrics, update_count, beta

    batch = replay_buffer.sample(batch_size, beta=beta)
    result = algorithm.update(batch, global_step=global_step)
    if algorithm.last_sequence_priorities is not None:
        replay_buffer.update_priorities(batch["indices"], algorithm.last_sequence_priorities)
    callback_list.on_update_end(trainer_state, result)
    return result.metrics, update_count + result.num_gradient_steps, beta


def train_agent57(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="agent57", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 10000))
    batch_size = int(config.algo_kwargs.get("batch_size", 32))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    target_update_interval = int(config.algo_kwargs.get("target_update_interval", 250))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-3))
    rnd_learning_rate = float(config.algo_kwargs.get("rnd_learning_rate", 1e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    sequence_length = int(config.algo_kwargs.get("sequence_length", 8))
    hidden_size = int(config.algo_kwargs.get("recurrent_hidden_size", 256))
    num_layers = int(config.algo_kwargs.get("recurrent_num_layers", 1))
    prioritized_alpha = float(config.algo_kwargs.get("prioritized_alpha", 0.6))
    prioritized_beta_start = float(config.algo_kwargs.get("prioritized_beta_start", 0.4))
    prioritized_beta_end = float(config.algo_kwargs.get("prioritized_beta_end", 1.0))
    prioritized_beta_fraction = float(config.algo_kwargs.get("prioritized_beta_fraction", 1.0))
    priority_eta = float(config.algo_kwargs.get("priority_eta", 0.9))
    n_step = int(config.algo_kwargs.get("n_step", 3))
    intrinsic_reward_coef = float(config.algo_kwargs.get("intrinsic_reward_coef", 0.1))
    eval_interval = resolve_eval_interval(config)
    prioritized_replay_schedule = _PrioritizedReplaySchedule(
        total_timesteps=config.total_timesteps,
        beta_start=prioritized_beta_start,
        beta_end=prioritized_beta_end,
        beta_fraction=prioritized_beta_fraction,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        if n_step <= 0:
            raise ValueError(f"n_step must be > 0, got {n_step}")

        obs_shape, action_dim = _infer_spaces(envs)
        q_network = _build_q_network(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        rnd_model = _build_rnd_model(config, obs_shape=obs_shape).to(device)
        algorithm = Agent57(
            q_network=q_network,
            rnd_model=rnd_model,
            learning_rate=learning_rate,
            rnd_learning_rate=rnd_learning_rate,
            gamma=gamma**n_step,
            target_update_interval=target_update_interval,
            double_q=True,
            priority_eta=priority_eta,
            intrinsic_reward_coef=intrinsic_reward_coef,
        )
        buffer_obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        replay_buffer = PrioritizedRecurrentReplayBuffer(
            capacity=buffer_capacity,
            num_envs=config.num_envs,
            obs_shape=obs_shape,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            alpha=prioritized_alpha,
            device=device,
            obs_dtype=buffer_obs_dtype,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        n_step_accumulator = NStepAccumulator(num_envs=config.num_envs, n_step=n_step, gamma=gamma)
        metadata_buffers: list[deque[dict[str, object]]] = [deque() for _ in range(config.num_envs)]

        obs, _ = envs.reset(seed=config.seed)
        recurrent_state = q_network.initial_state(config.num_envs, device=device)
        episode_starts = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        latest_update_metrics: MetricDict = {}
        latest_collect_metrics: MetricDict = {
            "extrinsic_reward_mean": 0.0,
            "intrinsic_reward_mean": 0.0,
            "combined_reward_mean": 0.0,
        }
        if checkpoint_state is not None:
            resume_context = checkpoint_state.trainer_state.get("resume_context")
            if isinstance(resume_context, dict):
                env_resume_state = resume_context.get("env_state")
                if isinstance(env_resume_state, dict):
                    restored_obs = restore_vector_env_resume_state(envs, env_resume_state)
                    if restored_obs is not None:
                        obs = np.asarray(restored_obs)
                random_state = resume_context.get("random_state")
                if isinstance(random_state, dict):
                    restore_global_random_state(random_state)
                n_step_state = resume_context.get("n_step_accumulator")
                if isinstance(n_step_state, dict):
                    n_step_accumulator.load_state_dict(n_step_state)
                metadata_buffers = _restore_metadata_buffers(
                    resume_context.get("metadata_buffers"),
                    num_envs=config.num_envs,
                    device=device,
                )
                recurrent_state, episode_starts = _restore_rollout_state(
                    payload=resume_context.get("rollout_state"),
                    initial_recurrent_state=recurrent_state,
                    num_envs=config.num_envs,
                    device=device,
                )
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            epsilon = resolve_exploration_epsilon(config, step=global_step)

            recurrent_state = q_network.reset_state(recurrent_state, episode_starts)
            state_snapshot = (recurrent_state[0].detach().clone(), recurrent_state[1].detach().clone())
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                rollout = q_network.act(obs_tensor, state=recurrent_state, epsilon=epsilon)

            next_obs, extrinsic_rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            with torch.no_grad():
                intrinsic_rewards = algorithm.intrinsic_reward(next_obs).detach().cpu().numpy().astype(np.float32)
            combined_rewards = np.asarray(extrinsic_rewards, dtype=np.float32) + intrinsic_reward_coef * intrinsic_rewards

            _append_recurrent_transitions(
                replay_buffer=replay_buffer,
                n_step_accumulator=n_step_accumulator,
                metadata_buffers=metadata_buffers,
                obs=obs,
                actions=rollout.actions,
                rewards=combined_rewards,
                next_obs=next_obs,
                dones=dones,
                episode_starts=episode_starts,
                state_snapshot=state_snapshot,
                num_envs=config.num_envs,
            )

            obs = next_obs
            recurrent_state = rollout.state
            episode_starts = torch.as_tensor(dones, dtype=torch.bool, device=device)
            global_step += config.num_envs
            trainer_state.global_step = global_step
            latest_collect_metrics = {
                "extrinsic_reward_mean": float(np.mean(extrinsic_rewards)),
                "intrinsic_reward_mean": float(np.mean(intrinsic_rewards)),
                "combined_reward_mean": float(np.mean(combined_rewards)),
            }
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                dones=dones,
                replay_buffer=replay_buffer,
                obs=obs,
                num_envs=config.num_envs,
            )

            latest_update_metrics, update_count, beta = _maybe_update_agent57(
                algorithm=algorithm,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_frequency=train_frequency,
                global_step=global_step,
                prioritized_replay_schedule=prioritized_replay_schedule,
                callback_list=callback_list,
                trainer_state=trainer_state,
                latest_update_metrics=latest_update_metrics,
                update_count=update_count,
            )
            trainer_state.update_count = update_count

            metrics = {
                **_build_r2d2_metrics(
                    latest_update_metrics,
                    epsilon=epsilon,
                    beta=beta,
                    global_step=global_step,
                    replay_buffer=replay_buffer,
                    update_count=update_count,
                ),
                **latest_collect_metrics,
                "intrinsic_reward_coef": intrinsic_reward_coef,
            }
            metrics, should_stop = _maybe_run_r2d2_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                q_network=q_network,
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
                    "n_step_accumulator": n_step_accumulator.state_dict(),
                    "metadata_buffers": _capture_metadata_buffers(metadata_buffers),
                    "rollout_state": _capture_rollout_state(recurrent_state, episode_starts),
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


__all__ = ["train_agent57", "_build_rnd_model", "_evaluate_r2d2_policy"]
