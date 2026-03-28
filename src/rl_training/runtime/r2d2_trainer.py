from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.r2d2 import R2D2
from rl_training.data.n_step import NStepAccumulator
from rl_training.data.prioritized_recurrent_replay_buffer import PrioritizedRecurrentReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.recurrent import LSTMQNetwork
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    resolve_eval_interval,
    resolve_exploration_epsilon,
    should_run_evaluation,
)
from rl_training.runtime.evaluation_support import evaluate_discrete_episodes
from rl_training.runtime.resume_state import (
    capture_global_random_state,
    capture_resume_value,
    capture_vector_env_resume_state,
    move_resume_value_to_device,
    restore_global_random_state,
    restore_resume_value,
    restore_vector_env_resume_state,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainerState, TrainResult
from rl_training.runtime.types import MetricDict


@dataclass(frozen=True)
class _PrioritizedReplaySchedule:
    total_timesteps: int
    beta_start: float
    beta_end: float
    beta_fraction: float


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for R2D2 trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for R2D2 trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(
            "expected flat 1D or channel-first image observations, "
            f"got shape={obs_space.shape!r}"
        )

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _epsilon_at_step(
    step: int,
    *,
    total_timesteps: int,
    epsilon_start: float,
    epsilon_end: float,
    exploration_fraction: float,
) -> float:
    decay_steps = max(1, int(total_timesteps * exploration_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def _beta_at_step(
    step: int,
    *,
    total_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_fraction: float,
) -> float:
    if beta_fraction <= 0:
        return float(beta_end)
    decay_steps = max(1, int(total_timesteps * beta_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(beta_start + progress * (beta_end - beta_start))


def _build_q_network(config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int) -> LSTMQNetwork:
    return LSTMQNetwork(
        obs_shape=obs_shape,
        action_dim=action_dim,
        features_dim=int(config.algo_kwargs.get("features_dim", 256)),
        encoder_hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (128,))),
        head_hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", (128,))),
        hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
        num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
    )


def _evaluate_r2d2_policy(
    q_network: LSTMQNetwork,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    class _ActionFn:
        def __init__(self) -> None:
            self.state: tuple[torch.Tensor, torch.Tensor] | None = None
            self.episode_starts: torch.Tensor | None = None

        def reset(self) -> None:
            self.state = q_network.initial_state(1, device=device)
            self.episode_starts = torch.ones(1, dtype=torch.bool, device=device)

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            if self.state is None or self.episode_starts is None:
                self.reset()
            with torch.no_grad():
                rollout = q_network.act(
                    obs_tensor,
                    state=self.state,
                    epsilon=0.0,
                    deterministic=True,
                    episode_starts=self.episode_starts,
                )
            self.state = rollout.state
            self.episode_starts = torch.zeros(1, dtype=torch.bool, device=device)
            action = rollout.actions.squeeze(0)
            return int(action.item())

    return evaluate_discrete_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=_ActionFn(),
    )


def _capture_rollout_state(
    recurrent_state: tuple[torch.Tensor, torch.Tensor],
    episode_starts: torch.Tensor,
) -> dict[str, object]:
    return {
        "recurrent_state": capture_resume_value(recurrent_state),
        "episode_starts": capture_resume_value(episode_starts),
    }


def _restore_rollout_state(
    *,
    payload: object,
    initial_recurrent_state: tuple[torch.Tensor, torch.Tensor],
    num_envs: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    recurrent_state = initial_recurrent_state
    episode_starts = torch.ones(num_envs, dtype=torch.bool, device=device)
    if not isinstance(payload, dict):
        return recurrent_state, episode_starts

    restored_state = payload.get("recurrent_state")
    if restored_state is not None:
        restored_recurrent_state = move_resume_value_to_device(
            restore_resume_value(restored_state),
            device=device,
        )
        if (
            isinstance(restored_recurrent_state, tuple)
            and len(restored_recurrent_state) == 2
            and torch.is_tensor(restored_recurrent_state[0])
            and torch.is_tensor(restored_recurrent_state[1])
        ):
            recurrent_state = (
                restored_recurrent_state[0].to(device=device),
                restored_recurrent_state[1].to(device=device),
            )

    restored_episode_starts = payload.get("episode_starts")
    if restored_episode_starts is not None:
        episode_start_tensor = move_resume_value_to_device(
            restore_resume_value(restored_episode_starts),
            device=device,
        )
        if torch.is_tensor(episode_start_tensor):
            episode_starts = episode_start_tensor.to(device=device, dtype=torch.bool)
    return recurrent_state, episode_starts


def _capture_metadata_buffers(metadata_buffers: list[deque[dict[str, object]]]) -> list[object]:
    return [capture_resume_value(list(buffer)) for buffer in metadata_buffers]


def _restore_metadata_buffers(
    payload: object,
    *,
    num_envs: int,
    device: torch.device,
) -> list[deque[dict[str, object]]]:
    restored_buffers: list[deque[dict[str, object]]] = [deque() for _ in range(num_envs)]
    if not isinstance(payload, list):
        return restored_buffers

    for env_index in range(min(num_envs, len(payload))):
        restored_buffer = move_resume_value_to_device(
            restore_resume_value(payload[env_index]),
            device=device,
        )
        if isinstance(restored_buffer, list):
            restored_buffers[env_index] = deque(
                item for item in restored_buffer if isinstance(item, dict)
            )
    return restored_buffers


def _append_recurrent_transitions(
    *,
    replay_buffer: PrioritizedRecurrentReplayBuffer,
    n_step_accumulator: NStepAccumulator,
    metadata_buffers: list[deque[dict[str, object]]],
    obs: np.ndarray,
    actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    episode_starts: torch.Tensor,
    state_snapshot: tuple[torch.Tensor, torch.Tensor],
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        metadata_buffers[env_index].append(
            {
                "episode_start": float(episode_starts[env_index].item()),
                "initial_state": (
                    state_snapshot[0][:, env_index : env_index + 1, :].clone(),
                    state_snapshot[1][:, env_index : env_index + 1, :].clone(),
                ),
            }
        )
        transitions = n_step_accumulator.add(
            env_index,
            obs[env_index],
            int(actions[env_index].item()),
            float(rewards[env_index]),
            next_obs[env_index],
            bool(dones[env_index]),
        )
        for transition in transitions:
            metadata = metadata_buffers[env_index].popleft()
            replay_buffer.add(
                env_index=env_index,
                obs=transition["obs"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                next_obs=transition["next_obs"],
                dones=transition["dones"],
                episode_start=metadata["episode_start"],
                initial_state=metadata["initial_state"],
            )


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    dones: np.ndarray,
    replay_buffer: PrioritizedRecurrentReplayBuffer,
    obs: np.ndarray,
    num_envs: int,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=num_envs,
            num_episodes=int(np.sum(dones)),
            metrics={
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "buffer_transitions": float(replay_buffer.num_transitions),
            },
            last_obs=obs,
        ),
    )


def _maybe_update_r2d2(
    *,
    algorithm: R2D2,
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


def _build_r2d2_metrics(
    latest_update_metrics: MetricDict,
    *,
    epsilon: float,
    beta: float,
    global_step: int,
    replay_buffer: PrioritizedRecurrentReplayBuffer,
    update_count: int,
) -> MetricDict:
    return {
        **latest_update_metrics,
        "epsilon": epsilon,
        "beta": beta,
        "global_step": float(global_step),
        "buffer_size": float(len(replay_buffer)),
        "buffer_transitions": float(replay_buffer.num_transitions),
        "gradient_steps": float(update_count),
    }


def _maybe_run_r2d2_evaluation(
    *,
    should_run_eval: bool,
    algorithm: R2D2,
    q_network: LSTMQNetwork,
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
    eval_metrics = _evaluate_r2d2_policy(
        q_network,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    algorithm.set_train_mode()
    evaluated_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(evaluated_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, evaluated_metrics)
    return evaluated_metrics, trainer_state.should_stop


def train_r2d2(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="r2d2", run_suffix=run_suffix, callbacks=callbacks)
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
        algorithm = R2D2(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma**n_step,
            target_update_interval=target_update_interval,
            double_q=True,
            priority_eta=priority_eta,
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

            next_obs, rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _append_recurrent_transitions(
                replay_buffer=replay_buffer,
                n_step_accumulator=n_step_accumulator,
                metadata_buffers=metadata_buffers,
                obs=obs,
                actions=rollout.actions,
                rewards=rewards,
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
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                dones=dones,
                replay_buffer=replay_buffer,
                obs=obs,
                num_envs=config.num_envs,
            )

            latest_update_metrics, update_count, beta = _maybe_update_r2d2(
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

            metrics = _build_r2d2_metrics(
                latest_update_metrics,
                epsilon=epsilon,
                beta=beta,
                global_step=global_step,
                replay_buffer=replay_buffer,
                update_count=update_count,
            )
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
