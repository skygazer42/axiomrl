from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.trpo import TRPO
from rl_training.data.rollout_buffer import RolloutBuffer
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import build_control_callbacks
from rl_training.runtime.ppo_trainer import _evaluate_policy
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict
from rl_training.envs.factory import make_vector_env


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for TRPO trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for TRPO trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"TRPO currently expects flat 1D observations, got shape={obs_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.n)


def train_trpo(
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
    trainer_state = TrainerState(algorithm="trpo", run_dir=run_context.run_dir)

    num_steps = int(config.algo_kwargs.get("num_steps", 128))
    value_updates = int(config.algo_kwargs.get("value_updates", 5))
    value_learning_rate = float(config.algo_kwargs.get("value_learning_rate", config.algo_kwargs.get("learning_rate", 1e-3)))
    max_kl = float(config.algo_kwargs.get("max_kl", 0.01))
    cg_iterations = int(config.algo_kwargs.get("cg_iterations", 10))
    cg_damping = float(config.algo_kwargs.get("cg_damping", 0.1))
    line_search_steps = int(config.algo_kwargs.get("line_search_steps", 10))
    line_search_shrink = float(config.algo_kwargs.get("line_search_shrink", 0.8))
    ent_coef = float(config.algo_kwargs.get("ent_coef", 0.0))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    gae_lambda = float(config.algo_kwargs.get("gae_lambda", 0.95))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_dim, action_dim = _infer_spaces(envs)
        policy = MLPActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(config.algo_kwargs.get("hidden_sizes", (64, 64))),
        ).to(device)
        algorithm = TRPO(
            policy=policy,
            value_learning_rate=value_learning_rate,
            max_kl=max_kl,
            cg_iterations=cg_iterations,
            cg_damping=cg_damping,
            line_search_steps=line_search_steps,
            line_search_shrink=line_search_shrink,
            value_updates=value_updates,
            ent_coef=ent_coef,
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
                obs_shape=(obs_dim,),
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

            result = algorithm.update(
                {
                    "obs": buffer.obs.reshape(num_steps * config.num_envs, obs_dim),
                    "actions": buffer.actions.reshape(num_steps * config.num_envs),
                    "logprobs": buffer.logprobs.reshape(num_steps * config.num_envs),
                    "advantages": buffer.advantages.reshape(num_steps * config.num_envs),
                    "returns": buffer.returns.reshape(num_steps * config.num_envs),
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
