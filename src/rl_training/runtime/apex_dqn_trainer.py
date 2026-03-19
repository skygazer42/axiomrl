from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.dqn import DoubleDQN
from rl_training.data.n_step import NStepAccumulator
from rl_training.data.prioritized_replay_buffer import PrioritizedReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn import CNNQNetwork
from rl_training.models.mlp_q_network import MLPQNetwork
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import resolve_eval_interval, should_run_evaluation
from rl_training.runtime.dqn_trainer import _evaluate_q_policy
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


@dataclass(frozen=True)
class _PrioritizedReplaySettings:
    total_timesteps: int
    beta_start: float
    beta_end: float
    beta_fraction: float


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for Ape-X DQN trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for Ape-X DQN trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")
    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


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


def _actor_epsilons(
    num_actors: int,
    *,
    base: float,
    alpha: float,
) -> torch.Tensor:
    if num_actors <= 0:
        raise ValueError(f"num_actors must be > 0, got {num_actors}")
    if not 0.0 <= float(base) <= 1.0:
        raise ValueError(f"actor_epsilon_base must be in [0, 1], got {base}")
    if float(alpha) < 0.0:
        raise ValueError(f"actor_epsilon_alpha must be >= 0, got {alpha}")

    if num_actors == 1:
        return torch.tensor([float(base)], dtype=torch.float32)

    # Ape-X commonly uses a geometric schedule across actors:
    #   eps_i = base ** (1 + (i/(N-1)) * alpha)
    # where base is in (0, 1) and alpha controls the spread.
    actor_ids = torch.arange(num_actors, dtype=torch.float32)
    exponents = 1.0 + (actor_ids / float(num_actors - 1)) * float(alpha)
    epsilons = float(base) ** exponents
    return epsilons.clamp(0.0, 1.0)


def _build_q_network(config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int) -> CNNQNetwork | MLPQNetwork:
    if len(obs_shape) == 3:
        head_hidden_sizes = tuple(
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))
        )
        return CNNQNetwork(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=head_hidden_sizes,
            features_dim=int(config.algo_kwargs.get("features_dim", 512)),
        )

    if len(obs_shape) != 1:
        raise ValueError(f"expected flat 1D or 3D image observations, got obs_shape={obs_shape!r}")
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    return MLPQNetwork(obs_dim=obs_shape[0], action_dim=action_dim, hidden_sizes=hidden_sizes)


def _select_actions(
    q_network: CNNQNetwork | MLPQNetwork,
    obs: np.ndarray,
    *,
    actor_epsilons: torch.Tensor,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        q_values = q_network(obs_tensor)
    greedy_actions = q_values.argmax(dim=-1)
    if actor_epsilons.max().item() <= 0.0:
        return greedy_actions

    eps = actor_epsilons.to(device=device).reshape(-1)
    if eps.shape[0] != greedy_actions.shape[0]:
        raise ValueError(
            f"actor_epsilons must have shape ({greedy_actions.shape[0]},), got {tuple(eps.shape)!r}"
        )
    random_actions = torch.randint(0, int(action_dim), greedy_actions.shape, device=device)
    explore_mask = torch.rand(greedy_actions.shape, device=device) < eps
    return torch.where(explore_mask, random_actions, greedy_actions)


def _store_transitions(
    *,
    replay_buffer: PrioritizedReplayBuffer,
    n_step_accumulator: NStepAccumulator,
    num_envs: int,
    obs: np.ndarray,
    actions: torch.Tensor,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
) -> None:
    for env_index in range(num_envs):
        action = int(actions[env_index].item())
        reward = float(rewards[env_index])
        done = bool(dones[env_index])
        for transition in n_step_accumulator.add(
            env_index,
            obs[env_index],
            action,
            reward,
            next_obs[env_index],
            done,
        ):
            replay_buffer.add(**transition)


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    num_envs: int,
    dones: np.ndarray,
    replay_buffer: PrioritizedReplayBuffer,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=num_envs,
            num_episodes=int(np.sum(dones)),
            metrics={
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
            },
            last_obs=obs,
        ),
    )


def _update_apex_dqn(
    *,
    algorithm: DoubleDQN,
    replay_buffer: PrioritizedReplayBuffer,
    batch_size: int,
    global_step: int,
    settings: _PrioritizedReplaySettings,
    updates_per_collect: int,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    latest_update_metrics: MetricDict,
    update_count: int,
) -> tuple[MetricDict, int, float]:
    if len(replay_buffer) < batch_size:
        return latest_update_metrics, update_count, _beta_at_step(
            global_step,
            total_timesteps=settings.total_timesteps,
            beta_start=settings.beta_start,
            beta_end=settings.beta_end,
            beta_fraction=settings.beta_fraction,
        )

    beta = _beta_at_step(
        global_step,
        total_timesteps=settings.total_timesteps,
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        beta_fraction=settings.beta_fraction,
    )
    metrics = latest_update_metrics
    gradient_steps = update_count
    for _ in range(int(updates_per_collect)):
        batch = replay_buffer.sample(batch_size, beta=beta)
        result = algorithm.update(batch, global_step=global_step)
        if algorithm.last_td_errors is not None:
            replay_buffer.update_priorities(batch["indices"], algorithm.last_td_errors)
        callback_list.on_update_end(trainer_state, result)
        metrics = result.metrics
        gradient_steps += result.num_gradient_steps
    return metrics, gradient_steps, beta


def train_apex_dqn(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="apex_dqn", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 512))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 50000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    target_update_interval = int(config.algo_kwargs.get("target_update_interval", 2500))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    n_step = int(config.algo_kwargs.get("n_step", 3))
    prioritized_alpha = float(config.algo_kwargs.get("prioritized_alpha", 0.6))
    prioritized_beta_start = float(config.algo_kwargs.get("prioritized_beta_start", 0.4))
    prioritized_beta_end = float(config.algo_kwargs.get("prioritized_beta_end", 1.0))
    prioritized_beta_fraction = float(config.algo_kwargs.get("prioritized_beta_fraction", 1.0))
    prioritized_eps = float(config.algo_kwargs.get("prioritized_eps", 1e-6))
    updates_per_collect = int(config.algo_kwargs.get("updates_per_collect", 1))
    actor_epsilon_base = float(config.algo_kwargs.get("actor_epsilon_base", 0.4))
    actor_epsilon_alpha = float(config.algo_kwargs.get("actor_epsilon_alpha", 7.0))
    eval_interval = resolve_eval_interval(config)

    prioritized_settings = _PrioritizedReplaySettings(
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
        if updates_per_collect <= 0:
            raise ValueError(f"updates_per_collect must be > 0, got {updates_per_collect}")
        if train_frequency <= 0:
            raise ValueError(f"train_frequency must be > 0, got {train_frequency}")

        obs_shape, action_dim = _infer_spaces(envs)
        actor_eps = _actor_epsilons(config.num_envs, base=actor_epsilon_base, alpha=actor_epsilon_alpha)

        q_network = _build_q_network(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        algorithm = DoubleDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=float(gamma) ** int(n_step),
            target_update_interval=target_update_interval,
        )

        obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            action_shape=(),
            alpha=prioritized_alpha,
            priority_eps=prioritized_eps,
            device=device,
            obs_dtype=obs_dtype,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        n_step_accumulator = NStepAccumulator(num_envs=config.num_envs, n_step=n_step, gamma=gamma)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)

        beta = _beta_at_step(
            global_step,
            total_timesteps=prioritized_settings.total_timesteps,
            beta_start=prioritized_settings.beta_start,
            beta_end=prioritized_settings.beta_end,
            beta_fraction=prioritized_settings.beta_fraction,
        )

        while global_step < config.total_timesteps:
            actions = _select_actions(
                q_network,
                obs,
                actor_epsilons=actor_eps,
                action_dim=action_dim,
                device=device,
            )
            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            _store_transitions(
                replay_buffer=replay_buffer,
                n_step_accumulator=n_step_accumulator,
                num_envs=config.num_envs,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
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

            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                latest_update_metrics, update_count, beta = _update_apex_dqn(
                    algorithm=algorithm,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    global_step=global_step,
                    settings=prioritized_settings,
                    updates_per_collect=updates_per_collect,
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
                "beta": float(beta),
                "actor_epsilon_mean": float(actor_eps.mean().item()),
                "actor_epsilon_max": float(actor_eps.max().item()),
                "actor_epsilon_min": float(actor_eps.min().item()),
            }

            metrics, should_stop = _maybe_run_apex_dqn_evaluation(
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


def _maybe_run_apex_dqn_evaluation(
    *,
    should_run_eval: bool,
    algorithm: DoubleDQN,
    q_network: CNNQNetwork | MLPQNetwork,
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
    eval_metrics = _evaluate_q_policy(
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
