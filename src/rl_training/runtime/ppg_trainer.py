from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.ppg import PPG
from rl_training.data.rollout_buffer import RolloutBuffer
from rl_training.envs.factory import build_env, make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.cnn import CNNPPGModel
from rl_training.models.mlp_ppg import MLPPPGModel
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
        raise TypeError(f"unsupported observation space for PPG trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for PPG trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
        raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _build_model(config: TrainConfig, *, obs_shape: tuple[int, ...], action_dim: int) -> MLPPPGModel | CNNPPGModel:
    if len(obs_shape) == 1:
        return MLPPPGModel(
            obs_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        )

    return CNNPPGModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_sizes=tuple(config.algo_kwargs.get("head_hidden_sizes", config.algo_kwargs.get("hidden_sizes", (512,)))),
        features_dim=int(config.algo_kwargs.get("features_dim", 512)),
    )


def _evaluate_ppg_policy(
    model: MLPPPGModel | CNNPPGModel,
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
                    action = model.act(obs_tensor, deterministic=True).actions.squeeze(0)
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


def _sample_auxiliary_minibatches(
    *,
    obs: torch.Tensor,
    returns: torch.Tensor,
    minibatch_size: int,
) -> list[dict[str, torch.Tensor]]:
    indices = torch.randperm(obs.shape[0], device=obs.device)
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, obs.shape[0], minibatch_size):
        batch_indices = indices[start : start + minibatch_size]
        batches.append({"obs": obs[batch_indices], "returns": returns[batch_indices]})
    return batches


def _restore_auxiliary_chunks(
    checkpoint_state: CheckpointState | None,
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if checkpoint_state is None or checkpoint_state.buffer_state is None:
        return [], []

    aux_obs_chunks = [torch.as_tensor(chunk, device=device) for chunk in checkpoint_state.buffer_state.get("aux_obs_chunks", [])]
    aux_return_chunks = [
        torch.as_tensor(chunk, dtype=torch.float32, device=device)
        for chunk in checkpoint_state.buffer_state.get("aux_return_chunks", [])
    ]
    return aux_obs_chunks, aux_return_chunks


def _collect_rollout(
    *,
    envs: gym.vector.SyncVectorEnv,
    buffer: RolloutBuffer,
    model: MLPPPGModel | CNNPPGModel,
    obs: np.ndarray,
    device: torch.device,
    num_steps: int,
    num_envs: int,
    buffer_obs_dtype: torch.dtype,
    trainer_state: TrainerState,
    global_step: int,
) -> tuple[np.ndarray, int]:
    current_obs = obs
    current_step = global_step
    for _ in range(num_steps):
        obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            rollout = model.act(obs_tensor)

        next_obs, rewards, terminated, truncated, _ = envs.step(rollout.actions.cpu().numpy())
        dones = np.logical_or(terminated, truncated).astype(np.float32)

        buffer.add(
            obs=current_obs if buffer_obs_dtype == torch.uint8 else obs_tensor,
            actions=rollout.actions,
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
            values=rollout.values,
            logprobs=rollout.logprobs,
        )

        current_obs = next_obs
        current_step += num_envs
        trainer_state.global_step = current_step

    return current_obs, current_step


def _append_auxiliary_rollout(
    buffer: RolloutBuffer,
    aux_obs_chunks: list[torch.Tensor],
    aux_return_chunks: list[torch.Tensor],
    *,
    num_steps: int,
    num_envs: int,
    obs_shape: tuple[int, ...],
    aux_buffer_rollouts: int,
) -> None:
    aux_obs_chunks.append(buffer.obs.reshape(num_steps * num_envs, *obs_shape).detach().clone())
    aux_return_chunks.append(buffer.returns.reshape(num_steps * num_envs).detach().clone())
    if len(aux_obs_chunks) > aux_buffer_rollouts:
        aux_obs_chunks.pop(0)
        aux_return_chunks.pop(0)


def _run_policy_updates(
    algorithm: PPG,
    buffer: RolloutBuffer,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    update_epochs: int,
    minibatch_size: int,
    global_step: int,
) -> tuple[MetricDict, int]:
    policy_metrics: MetricDict = {}
    policy_gradient_steps = 0
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
            policy_metrics = result.metrics
            policy_gradient_steps += result.num_gradient_steps
            callback_list.on_update_end(trainer_state, result)
    return policy_metrics, policy_gradient_steps


def _maybe_run_auxiliary_phase(
    algorithm: PPG,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    current_update: int,
    aux_frequency: int,
    aux_obs_chunks: list[torch.Tensor],
    aux_return_chunks: list[torch.Tensor],
    aux_epochs: int,
    aux_minibatch_size: int,
    global_step: int,
) -> tuple[MetricDict, int, float]:
    if not (aux_frequency > 0 and current_update % aux_frequency == 0 and aux_obs_chunks):
        return {}, 0, 0.0

    teacher_model = algorithm.snapshot_teacher_model()
    aux_obs = torch.cat(aux_obs_chunks, dim=0)
    aux_returns = torch.cat(aux_return_chunks, dim=0)
    auxiliary_metrics: MetricDict = {}
    auxiliary_gradient_steps = 0
    for _ in range(aux_epochs):
        for aux_batch in _sample_auxiliary_minibatches(
            obs=aux_obs,
            returns=aux_returns,
            minibatch_size=aux_minibatch_size,
        ):
            result = algorithm.auxiliary_update(
                aux_batch,
                teacher_model=teacher_model,
                global_step=global_step,
            )
            auxiliary_metrics = result.metrics
            auxiliary_gradient_steps += result.num_gradient_steps
            callback_list.on_update_end(trainer_state, result)
    return auxiliary_metrics, auxiliary_gradient_steps, 1.0


def _build_ppg_metrics(
    *,
    policy_metrics: MetricDict,
    auxiliary_metrics: MetricDict,
    eval_metrics: MetricDict,
    global_step: int,
    current_update: int,
    policy_gradient_steps: int,
    auxiliary_gradient_steps: int,
    auxiliary_phase_ran: float,
) -> MetricDict:
    return {
        **policy_metrics,
        **auxiliary_metrics,
        **eval_metrics,
        "global_step": float(global_step),
        "update": float(current_update),
        "gradient_steps": float(policy_gradient_steps + auxiliary_gradient_steps),
        "policy_gradient_steps": float(policy_gradient_steps),
        "auxiliary_gradient_steps": float(auxiliary_gradient_steps),
        "auxiliary_phase_ran": auxiliary_phase_ran,
    }


def train_ppg(
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
    trainer_state = TrainerState(algorithm="ppg", run_dir=run_context.run_dir)

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    update_epochs = int(config.algo_kwargs.get("update_epochs", 4))
    minibatch_size = int(config.algo_kwargs.get("minibatch_size", max(1, config.num_envs * num_steps // 4)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    aux_learning_rate = float(config.algo_kwargs.get("aux_learning_rate", learning_rate))
    clip_coef = resolve_clip_coefficient(config, step=0, default=0.2)
    ent_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="ent_coef", default=0.01)
    vf_coef = float(config.algo_kwargs.get("vf_coef", 0.5))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 0.5))
    aux_frequency = int(config.algo_kwargs.get("aux_frequency", 2))
    aux_epochs = int(config.algo_kwargs.get("aux_epochs", 2))
    aux_minibatch_size = int(config.algo_kwargs.get("aux_minibatch_size", minibatch_size))
    aux_buffer_rollouts = int(config.algo_kwargs.get("aux_buffer_rollouts", 4))
    aux_value_coef = float(config.algo_kwargs.get("aux_value_coef", 1.0))
    behavior_clone_coef = float(config.algo_kwargs.get("behavior_clone_coef", 1.0))
    value_clone_coef = float(config.algo_kwargs.get("value_clone_coef", 1.0))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_shape, action_dim = _infer_spaces(envs)
        model = _build_model(config, obs_shape=obs_shape, action_dim=action_dim).to(device)
        buffer_obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        algorithm = PPG(
            model=model,
            learning_rate=learning_rate,
            aux_learning_rate=aux_learning_rate,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            aux_value_coef=aux_value_coef,
            behavior_clone_coef=behavior_clone_coef,
            value_clone_coef=value_clone_coef,
            max_grad_norm=max_grad_norm,
        )

        aux_obs_chunks, aux_return_chunks = _restore_auxiliary_chunks(checkpoint_state, device=device)
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_index = int(checkpoint_state.trainer_state.get("update_index", 0)) if checkpoint_state is not None else 0
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

            obs, global_step = _collect_rollout(
                envs=envs,
                buffer=buffer,
                model=model,
                obs=obs,
                device=device,
                num_steps=num_steps,
                num_envs=config.num_envs,
                buffer_obs_dtype=buffer_obs_dtype,
                trainer_state=trainer_state,
                global_step=global_step,
            )

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
                last_values = model.act(torch.as_tensor(obs, dtype=torch.float32, device=device)).values

            buffer.compute_returns_and_advantages(
                last_values=last_values,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            _append_auxiliary_rollout(
                buffer,
                aux_obs_chunks,
                aux_return_chunks,
                num_steps=num_steps,
                num_envs=config.num_envs,
                obs_shape=obs_shape,
                aux_buffer_rollouts=aux_buffer_rollouts,
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
            policy_metrics, policy_gradient_steps = _run_policy_updates(
                algorithm,
                buffer,
                callback_list,
                trainer_state,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                global_step=global_step,
            )

            current_update = update_index + 1

            auxiliary_metrics, auxiliary_gradient_steps, auxiliary_phase_ran = _maybe_run_auxiliary_phase(
                algorithm,
                callback_list,
                trainer_state,
                current_update=current_update,
                aux_frequency=aux_frequency,
                aux_obs_chunks=aux_obs_chunks,
                aux_return_chunks=aux_return_chunks,
                aux_epochs=aux_epochs,
                aux_minibatch_size=aux_minibatch_size,
                global_step=global_step,
            )

            eval_metrics = _evaluate_ppg_policy(
                model,
                config,
                device=device,
                num_episodes=config.eval_episodes,
            )
            metrics = _build_ppg_metrics(
                policy_metrics=policy_metrics,
                auxiliary_metrics=auxiliary_metrics,
                eval_metrics=eval_metrics,
                global_step=global_step,
                current_update=current_update,
                policy_gradient_steps=policy_gradient_steps,
                auxiliary_gradient_steps=auxiliary_gradient_steps,
                auxiliary_phase_ran=auxiliary_phase_ran,
            )
            metrics["ent_coef"] = float(current_ent_coef)
            metrics["clip_coef"] = float(current_clip_coef)
            logger.log_metrics(metrics, step=global_step)
            callback_list.on_eval_end(trainer_state, metrics)
            update_index = current_update
            if trainer_state.should_stop:
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state={
                "aux_obs_chunks": [chunk.detach().cpu() for chunk in aux_obs_chunks],
                "aux_return_chunks": [chunk.detach().cpu() for chunk in aux_return_chunks],
            },
            trainer_state={
                "global_step": global_step,
                "update_index": update_index,
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
