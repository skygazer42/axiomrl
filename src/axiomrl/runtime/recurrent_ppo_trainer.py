from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from axiomrl.contrib.recurrent_ppo import RecurrentPPOAlgorithm
from axiomrl.data.recurrent_rollout_buffer import RecurrentRolloutBuffer
from axiomrl.envs.factory import make_vector_env
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.recurrent import LSTMActorCritic
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import resolve_clip_coefficient, resolve_entropy_coefficient
from axiomrl.runtime.evaluation_support import evaluate_discrete_episodes
from axiomrl.runtime.resume_state import (
    capture_global_random_state,
    capture_resume_value,
    capture_vector_env_resume_state,
    move_resume_value_to_device,
    restore_global_random_state,
    restore_resume_value,
    restore_vector_env_resume_state,
)
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainResult
from axiomrl.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.VectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for recurrent PPO trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for recurrent PPO trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _build_policy(config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int) -> LSTMActorCritic:
    if len(obs_shape) == 1:
        encoder_hidden_sizes = tuple(config.algo_kwargs.get("encoder_hidden_sizes", (128,)))
        head_hidden_sizes = tuple(config.algo_kwargs.get("head_hidden_sizes", (128,)))
    else:
        encoder_hidden_sizes = ()
        head_hidden_sizes = tuple(config.algo_kwargs.get("head_hidden_sizes", (128,)))

    return LSTMActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        features_dim=int(config.algo_kwargs.get("features_dim", 256)),
        encoder_hidden_sizes=encoder_hidden_sizes,
        head_hidden_sizes=head_hidden_sizes,
        hidden_size=int(config.algo_kwargs.get("recurrent_hidden_size", 256)),
        num_layers=int(config.algo_kwargs.get("recurrent_num_layers", 1)),
    )


def _evaluate_recurrent_policy(
    policy: LSTMActorCritic,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    class _ActionFn:
        def __init__(self) -> None:
            self.state: tuple[torch.Tensor, torch.Tensor] | None = None

        def reset(self) -> None:
            self.state = policy.initial_state(1, device=device)

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            if self.state is None:
                self.reset()
            with torch.no_grad():
                rollout = policy.act(obs_tensor, state=self.state, deterministic=True)
            self.state = rollout.state
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


def train_recurrent_ppo(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="recurrent_ppo", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    update_epochs = int(config.algo_kwargs.get("update_epochs", 4))
    minibatch_size = int(config.algo_kwargs.get("minibatch_size", max(1, config.num_envs * num_steps // 4)))
    sequence_length = int(config.algo_kwargs.get("sequence_length", min(num_steps, 16)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    clip_coef = resolve_clip_coefficient(config, step=0, default=0.2)
    ent_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="ent_coef", default=0.01)
    vf_coef = float(config.algo_kwargs.get("vf_coef", 0.5))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 0.5))
    hidden_size = int(config.algo_kwargs.get("recurrent_hidden_size", 256))
    num_layers = int(config.algo_kwargs.get("recurrent_num_layers", 1))
    sequences_per_batch = int(
        config.algo_kwargs.get("sequences_per_batch", max(1, minibatch_size // max(1, sequence_length)))
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = None
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        envs = make_vector_env(config)
        obs_shape, action_dim = _infer_spaces(envs)
        policy = _build_policy(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        algorithm = RecurrentPPOAlgorithm(
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
        recurrent_state = policy.initial_state(config.num_envs, device=device)
        episode_starts = torch.ones(config.num_envs, dtype=torch.bool, device=device)
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
                recurrent_state, episode_starts = _restore_rollout_state(
                    payload=resume_context.get("rollout_state"),
                    initial_recurrent_state=recurrent_state,
                    num_envs=config.num_envs,
                    device=device,
                )
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_index = int(checkpoint_state.trainer_state.get("update_index", 0)) if checkpoint_state is not None else 0
        trainer_state.global_step = global_step
        trainer_state.update_count = update_index
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            buffer = RecurrentRolloutBuffer(
                num_steps=num_steps,
                num_envs=config.num_envs,
                obs_shape=obs_shape,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
            )

            for _ in range(num_steps):
                recurrent_state = policy.reset_state(recurrent_state, episode_starts)
                state_snapshot = (recurrent_state[0].detach().clone(), recurrent_state[1].detach().clone())
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    rollout = policy.act(obs_tensor, state=recurrent_state)

                next_obs, rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated).astype(np.float32)
                done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
                buffer.add(
                    obs=obs_tensor,
                    actions=rollout.actions,
                    rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                    dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
                    episode_starts=episode_starts,
                    values=rollout.values,
                    logprobs=rollout.logprobs,
                    recurrent_state=state_snapshot,
                )

                recurrent_state = rollout.state
                obs = next_obs
                episode_starts = done_tensor
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

            recurrent_state = policy.reset_state(recurrent_state, episode_starts)
            with torch.no_grad():
                last_values = policy.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=device), state=recurrent_state
                ).values

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
                for minibatch in buffer.iter_sequence_minibatches(
                    sequence_length=sequence_length,
                    sequences_per_batch=sequences_per_batch,
                    shuffle=True,
                ):
                    result = algorithm.update(minibatch, global_step=global_step)
                    update_metrics = result.metrics
                    gradient_steps += result.num_gradient_steps
                    callback_list.on_update_end(trainer_state, result)

            eval_metrics = _evaluate_recurrent_policy(
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
            update_index += 1
            trainer_state.update_count = update_index
            if trainer_state.should_stop:
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
            trainer_state={
                "global_step": global_step,
                "update_index": update_index,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
                "resume_context": {
                    "env_state": capture_vector_env_resume_state(envs),
                    "random_state": capture_global_random_state(),
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
