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
from rl_training.models.mlp_ppg import MLPPPGModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for PPG trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for PPG trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.n)


def _evaluate_ppg_policy(
    model: MLPPPGModel,
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
    clip_coef = float(config.algo_kwargs.get("clip_coef", 0.2))
    ent_coef = float(config.algo_kwargs.get("ent_coef", 0.01))
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
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_dim, action_dim = _infer_spaces(envs)
        model = MLPPPGModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
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

        aux_obs_chunks: list[torch.Tensor] = []
        aux_return_chunks: list[torch.Tensor] = []
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                aux_obs_chunks = [
                    torch.as_tensor(chunk, dtype=torch.float32, device=device) for chunk in checkpoint_state.buffer_state.get("aux_obs_chunks", [])
                ]
                aux_return_chunks = [
                    torch.as_tensor(chunk, dtype=torch.float32, device=device)
                    for chunk in checkpoint_state.buffer_state.get("aux_return_chunks", [])
                ]

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_index = int(checkpoint_state.trainer_state.get("update_index", 0)) if checkpoint_state is not None else 0
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            buffer = RolloutBuffer(
                num_steps=num_steps,
                num_envs=config.num_envs,
                obs_shape=(obs_dim,),
                action_shape=(),
                device=device,
            )

            for _ in range(num_steps):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    rollout = model.act(obs_tensor)

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
                last_values = model.act(torch.as_tensor(obs, dtype=torch.float32, device=device)).values

            buffer.compute_returns_and_advantages(
                last_values=last_values,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            flattened_obs = buffer.obs.reshape(num_steps * config.num_envs, obs_dim).detach().clone()
            flattened_returns = buffer.returns.reshape(num_steps * config.num_envs).detach().clone()
            aux_obs_chunks.append(flattened_obs)
            aux_return_chunks.append(flattened_returns)
            if len(aux_obs_chunks) > aux_buffer_rollouts:
                aux_obs_chunks.pop(0)
                aux_return_chunks.pop(0)

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

            current_update = update_index + 1

            auxiliary_metrics: MetricDict = {}
            auxiliary_gradient_steps = 0
            auxiliary_phase_ran = 0.0
            if aux_frequency > 0 and current_update % aux_frequency == 0 and aux_obs_chunks:
                teacher_model = algorithm.snapshot_teacher_model()
                aux_obs = torch.cat(aux_obs_chunks, dim=0)
                aux_returns = torch.cat(aux_return_chunks, dim=0)
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
                auxiliary_phase_ran = 1.0

            eval_metrics = _evaluate_ppg_policy(
                model,
                config,
                device=device,
                num_episodes=config.eval_episodes,
            )
            metrics = {
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
