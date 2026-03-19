from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.mbpo import MBPO
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.controls import resolve_eval_interval, should_run_periodic_eval
from rl_training.runtime.off_policy_trainer_utils import emit_collect_event, store_vector_transitions
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.sac_trainer import _action_bounds, _evaluate_sac_policy, _scale_actions
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for MBPO trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for MBPO trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
    if action_space.shape is None or len(action_space.shape) != 1:
        raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.shape[0])


def _concatenate_transition_batches(batches: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("at least one batch is required")
    if len(batches) == 1:
        return dict(batches[0])

    merged: dict[str, torch.Tensor] = {}
    common_keys = set(batches[0]).intersection(*(set(batch) for batch in batches[1:]))
    for key in common_keys:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    return merged


def _sample_mixed_batch(
    replay_buffer: ReplayBuffer,
    synthetic_buffer: ReplayBuffer,
    *,
    batch_size: int,
    synthetic_batch_ratio: float,
) -> dict[str, torch.Tensor]:
    synthetic_fraction = float(synthetic_batch_ratio)
    if synthetic_fraction <= 0.0 or len(synthetic_buffer) == 0:
        return replay_buffer.sample(batch_size)

    synthetic_size = int(round(batch_size * synthetic_fraction))
    synthetic_size = max(0, min(batch_size, synthetic_size))
    real_size = batch_size - synthetic_size

    batches: list[dict[str, torch.Tensor]] = []
    if real_size > 0:
        batches.append(replay_buffer.sample(real_size))
    if synthetic_size > 0:
        batches.append(synthetic_buffer.sample(synthetic_size))

    return _concatenate_transition_batches(batches)


def _refresh_synthetic_buffer(
    *,
    algorithm: MBPO,
    replay_buffer: ReplayBuffer,
    synthetic_buffer: ReplayBuffer,
    rollout_batch_size: int,
    rollout_horizon: int,
    device: torch.device,
) -> MetricDict:
    synthetic_buffer.reset()
    if rollout_batch_size < 1 or rollout_horizon < 1 or len(replay_buffer) == 0:
        return {
            "synthetic_buffer_size": float(len(synthetic_buffer)),
            "synthetic_rollout_transitions": 0.0,
            "synthetic_reward_mean": 0.0,
            "synthetic_disagreement_mean": 0.0,
        }

    algorithm.set_eval_mode()
    reward_means: list[float] = []
    disagreement_means: list[float] = []
    transitions_added = 0

    current_obs = replay_buffer.sample(rollout_batch_size)["obs"]
    with torch.no_grad():
        for _ in range(rollout_horizon):
            actions = algorithm.policy_model.sample_actions(current_obs).actions
            synthetic = algorithm.sample_synthetic_transition(current_obs, actions)
            dones = torch.zeros(current_obs.shape[0], dtype=torch.float32, device=device)

            for index in range(int(current_obs.shape[0])):
                synthetic_buffer.add(
                    obs=current_obs[index],
                    actions=actions[index],
                    rewards=synthetic["rewards"][index],
                    next_obs=synthetic["next_obs"][index],
                    dones=dones[index],
                )

            reward_means.append(float(synthetic["rewards"].mean().detach().cpu().item()))
            disagreement_means.append(float(synthetic["disagreement"].mean().detach().cpu().item()))
            transitions_added += int(current_obs.shape[0])
            current_obs = synthetic["next_obs"]

    algorithm.set_train_mode()
    return {
        "synthetic_buffer_size": float(len(synthetic_buffer)),
        "synthetic_rollout_transitions": float(transitions_added),
        "synthetic_reward_mean": float(np.mean(reward_means)) if reward_means else 0.0,
        "synthetic_disagreement_mean": float(np.mean(disagreement_means)) if disagreement_means else 0.0,
    }


def train_mbpo(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="mbpo", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    synthetic_buffer_capacity = int(config.algo_kwargs.get("synthetic_buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    model_batch_size = int(config.algo_kwargs.get("model_batch_size", batch_size))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    model_train_frequency = int(config.algo_kwargs.get("model_train_frequency", train_frequency))
    model_updates = int(config.algo_kwargs.get("model_updates", 1))

    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    model_hidden_sizes = tuple(config.algo_kwargs.get("model_hidden_sizes", hidden_sizes))
    num_ensembles = int(config.algo_kwargs.get("num_ensembles", 5))
    policy_learning_rate = float(config.algo_kwargs.get("policy_learning_rate", 3e-4))
    model_learning_rate = float(config.algo_kwargs.get("model_learning_rate", 1e-3))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    alpha = float(config.algo_kwargs.get("alpha", 0.2))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    eval_interval = resolve_eval_interval(config)

    rollout_batch_size = int(config.algo_kwargs.get("rollout_batch_size", 1024))
    rollout_horizon = int(config.algo_kwargs.get("rollout_horizon", 1))
    rollout_refresh_interval = int(config.algo_kwargs.get("rollout_refresh_interval", 250))
    synthetic_batch_ratio = float(config.algo_kwargs.get("synthetic_batch_ratio", 0.5))

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
            raise TypeError(f"unsupported action space for MBPO trainer: {type(action_space)!r}")
        low, high = _action_bounds(action_space, device=device)

        policy_model = MLPSACModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        dynamics_model = MLPMOPOEnsembleModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_hidden_sizes,
            num_ensembles=num_ensembles,
        ).to(device)
        algorithm = MBPO(
            policy_model=policy_model,
            dynamics_model=dynamics_model,
            policy_learning_rate=policy_learning_rate,
            model_learning_rate=model_learning_rate,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
        )

        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=device,
        )
        synthetic_buffer = ReplayBuffer(
            capacity=synthetic_buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=device,
        )

        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_state = checkpoint_state.buffer_state.get("replay_buffer")
                synthetic_state = checkpoint_state.buffer_state.get("synthetic_buffer")
                if replay_state is not None:
                    replay_buffer.load_state_dict(replay_state)
                if synthetic_state is not None:
                    synthetic_buffer.load_state_dict(synthetic_state)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        model_update_count = int(checkpoint_state.trainer_state.get("model_update_count", 0)) if checkpoint_state is not None else 0
        latest_update_metrics: MetricDict = {}
        latest_model_metrics: MetricDict = {}
        latest_refresh_metrics: MetricDict = {
            "synthetic_buffer_size": float(len(synthetic_buffer)),
            "synthetic_rollout_transitions": 0.0,
            "synthetic_reward_mean": 0.0,
            "synthetic_disagreement_mean": 0.0,
        }

        trainer_state.global_step = global_step
        trainer_state.update_count = update_count + model_update_count
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                normalized_actions = policy_model.sample_actions(obs_tensor).actions
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

            if (
                model_updates > 0
                and model_train_frequency > 0
                and len(replay_buffer) >= max(model_batch_size, learning_starts)
                and global_step % model_train_frequency == 0
            ):
                for _ in range(model_updates):
                    result = algorithm.update_model(replay_buffer.sample(model_batch_size), global_step=global_step)
                    latest_model_metrics = result.metrics
                    model_update_count += result.num_gradient_steps
                    trainer_state.update_count = update_count + model_update_count
                    callback_list.on_update_end(trainer_state, result)

            if (
                rollout_refresh_interval > 0
                and len(replay_buffer) >= max(rollout_batch_size, learning_starts)
                and (len(synthetic_buffer) == 0 or global_step % rollout_refresh_interval == 0)
            ):
                latest_refresh_metrics = _refresh_synthetic_buffer(
                    algorithm=algorithm,
                    replay_buffer=replay_buffer,
                    synthetic_buffer=synthetic_buffer,
                    rollout_batch_size=rollout_batch_size,
                    rollout_horizon=rollout_horizon,
                    device=device,
                )

            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                batch = _sample_mixed_batch(
                    replay_buffer,
                    synthetic_buffer,
                    batch_size=batch_size,
                    synthetic_batch_ratio=synthetic_batch_ratio,
                )
                result = algorithm.update(batch, global_step=global_step)
                latest_update_metrics = result.metrics
                update_count += result.num_gradient_steps
                trainer_state.update_count = update_count + model_update_count
                callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                **latest_model_metrics,
                **latest_refresh_metrics,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "synthetic_buffer_size": float(len(synthetic_buffer)),
                "gradient_steps": float(update_count),
                "model_gradient_steps": float(model_update_count),
                "alpha": float(alpha),
                "synthetic_batch_ratio": float(synthetic_batch_ratio),
            }

            if should_run_periodic_eval(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_sac_policy(
                    policy_model,
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
            buffer_state={
                "replay_buffer": replay_buffer.state_dict(),
                "synthetic_buffer": synthetic_buffer.state_dict(),
            },
            trainer_state={
                "global_step": global_step,
                "update_count": update_count,
                "model_update_count": model_update_count,
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
