from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.discrete_sac import DiscreteSAC
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_discrete_sac import MLPDiscreteSACModel
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import resolve_eval_interval, should_run_periodic_eval
from rl_training.runtime.evaluation_support import evaluate_discrete_episodes
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for Discrete SAC trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for Discrete SAC trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"Discrete SAC currently expects flat 1D observations, got shape={obs_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.n)


def _evaluate_discrete_sac_policy(
    model: MLPDiscreteSACModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    def action_fn(obs_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            action = model.sample_actions(obs_tensor, deterministic=True).actions.squeeze(0)
        return int(action.item())

    return evaluate_discrete_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=action_fn,
    )


def train_discrete_sac(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(
        config,
        algorithm="discrete_sac",
        run_suffix=run_suffix,
        callbacks=callbacks,
    )
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
    alpha = float(config.algo_kwargs.get("alpha", 0.2))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_dim, action_dim = _infer_spaces(envs)
        model = MLPDiscreteSACModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = DiscreteSAC(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            alpha=alpha,
            tau=tau,
        )
        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=(obs_dim,),
            action_shape=(),
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
                actions = model.sample_actions(obs_tensor).actions

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            for env_index in range(config.num_envs):
                replay_buffer.add(
                    obs=obs[env_index],
                    actions=actions[env_index],
                    rewards=float(rewards[env_index]),
                    next_obs=next_obs[env_index],
                    dones=float(dones[env_index]),
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

            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)
                latest_update_metrics = result.metrics
                update_count += result.num_gradient_steps
                callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "alpha": alpha,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
            }
            if should_run_periodic_eval(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_discrete_sac_policy(
                    model,
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
