from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.drqn import DRQN
from rl_training.data.recurrent_replay_buffer import RecurrentReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.recurrent import LSTMQNetwork
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    resolve_eval_interval,
    resolve_exploration_epsilon,
    should_run_evaluation,
)
from rl_training.runtime.evaluation_support import evaluate_discrete_episodes
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for DRQN trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for DRQN trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"DRQN v1 currently supports flat 1D observations only, got shape={obs_space.shape!r}")

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


def _evaluate_drqn_policy(
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


def train_drqn(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="drqn", run_suffix=run_suffix, callbacks=callbacks)
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
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_shape, action_dim = _infer_spaces(envs)
        q_network = _build_q_network(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        algorithm = DRQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        replay_buffer = RecurrentReplayBuffer(
            capacity=buffer_capacity,
            num_envs=config.num_envs,
            obs_shape=obs_shape,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)
                replay_buffer.clear_active_chunks()

        obs, _ = envs.reset(seed=config.seed)
        recurrent_state = q_network.initial_state(config.num_envs, device=device)
        episode_starts = torch.ones(config.num_envs, dtype=torch.bool, device=device)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        latest_update_metrics: MetricDict = {}
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

            for env_index in range(config.num_envs):
                replay_buffer.add(
                    env_index=env_index,
                    obs=obs[env_index],
                    actions=int(rollout.actions[env_index].item()),
                    rewards=float(rewards[env_index]),
                    next_obs=next_obs[env_index],
                    dones=float(dones[env_index]),
                    episode_start=float(episode_starts[env_index].item()),
                    initial_state=(
                        state_snapshot[0][:, env_index : env_index + 1, :],
                        state_snapshot[1][:, env_index : env_index + 1, :],
                    ),
                )

            obs = next_obs
            recurrent_state = rollout.state
            episode_starts = torch.as_tensor(dones, dtype=torch.bool, device=device)
            global_step += config.num_envs
            trainer_state.global_step = global_step
            callback_list.on_collect_end(
                trainer_state,
                CollectResult(
                    num_env_steps=config.num_envs,
                    num_episodes=int(np.sum(dones)),
                    metrics={
                        "global_step": float(global_step),
                        "buffer_size": float(len(replay_buffer)),
                        "buffer_transitions": float(replay_buffer.num_transitions),
                    },
                    last_obs=obs,
                ),
            )

            if (
                replay_buffer.num_transitions >= learning_starts
                and len(replay_buffer) >= batch_size
                and global_step % train_frequency == 0
            ):
                result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
                latest_update_metrics = result.metrics
                update_count += result.num_gradient_steps
                trainer_state.update_count = update_count
                callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "epsilon": epsilon,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "buffer_transitions": float(replay_buffer.num_transitions),
                "gradient_steps": float(update_count),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_drqn_policy(
                    q_network,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                )
                algorithm.set_train_mode()
                metrics = {**metrics, **eval_metrics}
                logger.log_metrics(metrics, step=global_step)
                callback_list.on_eval_end(trainer_state, metrics)
                if trainer_state.should_stop:
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
