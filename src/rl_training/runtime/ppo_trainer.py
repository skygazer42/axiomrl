from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.ppo import PPO
from rl_training.data.rollout_buffer import RolloutBuffer
from rl_training.envs.factory import build_env, make_vector_env
from rl_training.experiment.checkpointing import CheckpointState, save_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn import CNNActorCritic
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks, resolve_clip_coefficient, resolve_entropy_coefficient
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for PPO trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for PPO trainer: {type(action_space)!r}")
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
    env = build_env(config, 0, evaluation=True)
    returns: list[float] = []

    try:
        for episode_index in range(num_episodes):
            obs, _ = env.reset(seed=config.seed + episode_index)
            done = False
            truncated = False
            episode_return = 0.0

            while not (done or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action = policy.act(obs_tensor, deterministic=True).actions.squeeze(0)
                obs, reward, done, truncated, _ = env.step(int(action.item()))
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_episodes": float(len(returns)),
    }


def train_ppo(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    device = resolve_device(config.device)
    run_artifacts = create_training_run(config, run_suffix=run_suffix)
    run_context = run_artifacts.run_context
    logger = run_artifacts.logger
    callback_list = CallbackList(merge_callbacks(build_control_callbacks(config), callbacks))
    trainer_state = TrainerState(algorithm="ppo", run_dir=run_context.run_dir)

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    update_epochs = int(config.algo_kwargs.get("update_epochs", 4))
    minibatch_size = int(config.algo_kwargs.get("minibatch_size", max(1, config.num_envs * num_steps // 4)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    clip_coef = resolve_clip_coefficient(config, step=0, default=0.2)
    ent_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="ent_coef", default=0.01)
    vf_coef = float(config.algo_kwargs.get("vf_coef", 0.5))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 0.5))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_shape, action_dim = _infer_spaces(envs)
        policy = _build_policy(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        algorithm = PPO(
            policy=policy,
            learning_rate=learning_rate,
            clip_coef=clip_coef,
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
            )

            for _ in range(num_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    rollout = policy.act(obs_tensor)

                next_obs, rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated).astype(np.float32)

                buffer.add(
                    obs=obs_tensor,
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
            current_clip_coef = resolve_clip_coefficient(config, step=global_step, default=0.2)
            algorithm.ent_coef = current_ent_coef
            algorithm.clip_coef = current_clip_coef
            update_metrics: MetricDict = {}
            gradient_steps = 0
            for _ in range(update_epochs):
                for minibatch in buffer.iter_minibatches(minibatch_size=minibatch_size, shuffle=True):
                    result = algorithm.update(
                        {
                            "obs": minibatch["obs"],
                            "actions": minibatch["actions"],
                            "logprobs": minibatch["logprobs"],
                            "advantages": minibatch["advantages"],
                            "returns": minibatch["returns"],
                        },
                        global_step=global_step,
                    )
                    update_metrics = result.metrics
                    gradient_steps += result.num_gradient_steps
                    callback_list.on_update_end(trainer_state, result)

            eval_metrics = _evaluate_policy(
                policy,
                config,
                device=device,
                num_episodes=config.eval_episodes,
            )
            metrics = {
                **update_metrics,
                **eval_metrics,
                "global_step": float(global_step),
                "update": float(update_index + 1),
                "gradient_steps": float(gradient_steps),
                "ent_coef": float(current_ent_coef),
                "clip_coef": float(current_clip_coef),
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
        envs.close()
        run_artifacts.close()
    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
