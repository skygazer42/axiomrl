from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from axiomrl.algorithms.gail import GAIL
from axiomrl.data.dataset_loaders import load_transition_dataset
from axiomrl.data.offline_dataset import TransitionDataset
from axiomrl.data.rollout_buffer import RolloutBuffer
from axiomrl.envs.factory import build_env, make_vector_env
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.cnn import CNNActorCritic
from axiomrl.models.mlp_actor_critic import MLPActorCritic
from axiomrl.models.mlp_gail_discriminator import CNNGAILDiscriminator, MLPGAILDiscriminator
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import resolve_clip_coefficient, resolve_entropy_coefficient
from axiomrl.runtime.evaluation_support import evaluate_discrete_episodes
from axiomrl.runtime.resume_state import (
    capture_global_random_state,
    capture_vector_env_resume_state,
    restore_global_random_state,
    restore_vector_env_resume_state,
)
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainResult
from axiomrl.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for GAIL trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for GAIL trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _build_policy(
    config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int
) -> MLPActorCritic | CNNActorCritic:
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


def _build_discriminator(
    config: TrainConfig,
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
) -> MLPGAILDiscriminator | CNNGAILDiscriminator:
    if len(obs_shape) == 1:
        hidden_sizes = tuple(
            config.algo_kwargs.get("discriminator_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (64, 64)))
        )
        return MLPGAILDiscriminator(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )

    hidden_sizes = tuple(
        config.algo_kwargs.get(
            "discriminator_head_hidden_sizes",
            config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,))),
        )
    )
    return CNNGAILDiscriminator(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        features_dim=int(
            config.algo_kwargs.get("discriminator_features_dim", config.algo_kwargs.get("features_dim", 512))
        ),
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


def _collect_random_expert_dataset(config: TrainConfig, *, dataset_size: int, seed: int) -> TransitionDataset:
    env = build_env(config, 0, evaluation=False)
    try:
        obs, _ = env.reset(seed=seed)
        obs_parts: list[Any] = []
        action_parts: list[Any] = []
        reward_parts: list[float] = []
        next_obs_parts: list[Any] = []
        done_parts: list[float] = []

        for step in range(int(dataset_size)):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            obs_parts.append(obs)
            action_parts.append(action)
            reward_parts.append(float(reward))
            next_obs_parts.append(next_obs)
            done_parts.append(float(done))

            if done:
                obs, _ = env.reset(seed=seed + step + 1)
            else:
                obs = next_obs

        return TransitionDataset.from_dict(
            {
                "obs": np.asarray(obs_parts),
                "actions": np.asarray(action_parts),
                "rewards": np.asarray(reward_parts, dtype=np.float32),
                "next_obs": np.asarray(next_obs_parts),
                "dones": np.asarray(done_parts, dtype=np.float32),
            }
        )
    finally:
        env.close()


def _build_expert_dataset(config: TrainConfig) -> TransitionDataset:
    kind = str(config.algo_kwargs.get("expert_dataset_kind", "random")).lower()

    if kind == "random":
        dataset_size = int(config.algo_kwargs.get("expert_dataset_size", 10000))
        dataset_seed = int(config.algo_kwargs.get("expert_dataset_seed", config.seed))
        return _collect_random_expert_dataset(config, dataset_size=dataset_size, seed=dataset_seed)

    if kind in {"npz", "minari"}:
        return load_transition_dataset(
            kind,
            dataset_path=config.algo_kwargs.get("expert_dataset_path"),
            dataset_id=config.algo_kwargs.get("expert_dataset_id"),
            download=bool(config.algo_kwargs.get("expert_dataset_download", False)),
        )

    raise ValueError(f"unsupported expert_dataset_kind: {kind!r}")


def _sample_policy_transitions(buffer: RolloutBuffer, batch_size: int) -> dict[str, torch.Tensor]:
    total_items = buffer.num_steps * buffer.num_envs
    indices = torch.randint(0, total_items, (int(batch_size),), device=buffer.device)
    obs = buffer.obs.reshape(total_items, *buffer.obs_shape).index_select(0, indices)
    actions = buffer.actions.reshape(total_items).index_select(0, indices)
    return {"obs": obs, "actions": actions}


def train_gail(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="gail", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    update_epochs = int(config.algo_kwargs.get("update_epochs", 4))
    minibatch_size = int(config.algo_kwargs.get("minibatch_size", 256))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    clip_coef = resolve_clip_coefficient(config, step=0, default=0.2)
    ent_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="ent_coef", default=0.01)
    vf_coef = float(config.algo_kwargs.get("vf_coef", 0.5))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 0.5))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))

    discriminator_learning_rate = float(config.algo_kwargs.get("discriminator_learning_rate", learning_rate))
    discriminator_updates = int(config.algo_kwargs.get("discriminator_updates", 4))
    discriminator_batch_size = int(config.algo_kwargs.get("discriminator_batch_size", minibatch_size))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_shape, action_dim = _infer_spaces(envs)
        policy = _build_policy(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        discriminator = _build_discriminator(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        algorithm = GAIL(
            policy=policy,
            discriminator=discriminator,
            learning_rate=learning_rate,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            discriminator_learning_rate=discriminator_learning_rate,
            max_grad_norm=max_grad_norm,
        )

        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        expert_dataset = _build_expert_dataset(config)

        obs, _ = envs.reset(seed=config.seed)
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
                    gail_rewards = algorithm.discriminator_reward(obs_tensor, rollout.actions).detach()

                next_obs, _, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated).astype(np.float32)

                buffer.add(
                    obs=obs_tensor,
                    actions=rollout.actions,
                    rewards=gail_rewards,
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

            discriminator_metrics: MetricDict = {}
            for _ in range(max(0, discriminator_updates)):
                expert_batch = expert_dataset.sample(discriminator_batch_size, device=device)
                policy_batch = _sample_policy_transitions(buffer, discriminator_batch_size)
                result = algorithm.update_discriminator(policy_batch, expert_batch, global_step=global_step)
                discriminator_metrics = result.metrics
                callback_list.on_update_end(trainer_state, result)

            current_ent_coef = resolve_entropy_coefficient(
                config,
                step=global_step,
                coefficient_key="ent_coef",
                default=0.01,
            )
            current_clip_coef = resolve_clip_coefficient(config, step=global_step, default=0.2)
            algorithm.ppo.ent_coef = current_ent_coef
            algorithm.ppo.clip_coef = current_clip_coef
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
                **discriminator_metrics,
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
                "resume_context": {
                    "env_state": capture_vector_env_resume_state(envs),
                    "random_state": capture_global_random_state(),
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
