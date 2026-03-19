from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.a2c import A2C
from rl_training.data.rollout_buffer import RolloutBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn import CNNActorCritic
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import resolve_entropy_coefficient
from rl_training.runtime.evaluation_support import evaluate_discrete_episodes
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for A2C trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for A2C trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _build_policy(config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int) -> MLPActorCritic | CNNActorCritic:
    if len(obs_shape) == 1:
        hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
        return MLPActorCritic(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

    return CNNActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))),
        features_dim=int(config.algo_kwargs.get("features_dim", 512)),
    )


def _evaluate_policy(
    policy: MLPActorCritic | CNNActorCritic,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    def action_fn(obs_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            action = policy.act(obs_tensor, deterministic=True).actions.squeeze(0)
        return int(action.item())

    return evaluate_discrete_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=action_fn,
    )


def train_a2c(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="a2c", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    ent_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="ent_coef", default=0.01)
    vf_coef = float(config.algo_kwargs.get("vf_coef", 0.5))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 0.5))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_shape, action_dim = _infer_spaces(envs)
        policy = _build_policy(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        buffer_obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        algorithm = A2C(
            policy=policy,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )

        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_index = 0
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            buffer = RolloutBuffer(
                num_steps=num_steps,
                num_envs=config.num_envs,
                obs_shape=obs_shape,
                action_shape=(),
                device=device,
                obs_dtype=buffer_obs_dtype,
            )

            for _ in range(num_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    rollout = policy.act(obs_tensor)

                next_obs, rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated).astype(np.float32)

                buffer.add(
                    obs=obs if buffer_obs_dtype == torch.uint8 else obs_tensor,
                    actions=rollout.actions,
                    rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                    dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
                    values=rollout.values,
                    logprobs=rollout.logprobs,
                )

                obs = next_obs
                global_step += config.num_envs
                trainer_state.global_step = global_step

            callback_list.on_collect_end(
                trainer_state,
                CollectResult(
                    num_env_steps=num_steps * config.num_envs,
                    num_episodes=int(buffer.dones.sum().item()),
                    metrics={"global_step": float(global_step)},
                    last_obs=obs,
                ),
            )

            with torch.no_grad():
                last_values = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=device)).values

            buffer.compute_returns_and_advantages(
                last_values=last_values,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            current_ent_coef = resolve_entropy_coefficient(
                config,
                step=global_step,
                coefficient_key="ent_coef",
                default=0.01,
            )
            algorithm.ent_coef = current_ent_coef
            batch = next(buffer.iter_minibatches(minibatch_size=num_steps * config.num_envs, shuffle=False))
            result = algorithm.update(
                {
                    "obs": batch["obs"],
                    "actions": batch["actions"],
                    "advantages": batch["advantages"],
                    "returns": batch["returns"],
                },
                global_step=global_step,
            )
            callback_list.on_update_end(trainer_state, result)

            eval_metrics = _evaluate_policy(
                policy,
                config,
                device=device,
                num_episodes=config.eval_episodes,
            )
            metrics = {
                **result.metrics,
                **eval_metrics,
                "global_step": float(global_step),
                "update": float(update_index + 1),
                "gradient_steps": float(result.num_gradient_steps),
                "ent_coef": float(current_ent_coef),
            }
            logger.log_metrics(metrics, step=global_step)
            callback_list.on_eval_end(trainer_state, metrics)
            if trainer_state.should_stop:
                break
            update_index += 1

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
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
