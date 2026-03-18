from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.dreamer import Dreamer
from rl_training.algorithms.eadream import EADream
from rl_training.algorithms.diamond import Diamond
from rl_training.algorithms.horizon_imagination import HorizonImagination
from rl_training.algorithms.po_dreamer import PODreamer
from rl_training.algorithms.twisted import Twisted
from rl_training.algorithms.dreamerv3 import DreamerV3
from rl_training.algorithms.mow import MoW
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import build_env, make_vector_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.dreamer import DreamerModel
from rl_training.models.eadream import EADreamModel
from rl_training.models.mow import MoWModel
from rl_training.models.po_dreamer import PODreamerModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.controls import (
    build_control_callbacks,
    resolve_entropy_coefficient,
    resolve_eval_interval,
    should_run_periodic_eval,
)
from rl_training.runtime.off_policy_trainer_utils import emit_collect_event, store_vector_transitions
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[tuple[int, ...], int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for Dreamer trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for Dreamer trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 3:
        raise ValueError(f"expected channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _evaluate_policy(
    model: DreamerModel,
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


def train_dreamer(
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
    trainer_state = TrainerState(algorithm=config.algo, run_dir=run_context.run_dir)

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 100000))
    batch_size = int(config.algo_kwargs.get("batch_size", 32))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    world_model_updates = int(config.algo_kwargs.get("world_model_updates", 1))
    actor_critic_updates = int(config.algo_kwargs.get("actor_critic_updates", 1))
    imagination_batch_size = int(config.algo_kwargs.get("imagination_batch_size", batch_size))
    imagination_horizon = int(config.algo_kwargs.get("imagination_horizon", 5))

    features_dim = int(config.algo_kwargs.get("features_dim", 128))
    action_embed_dim = int(config.algo_kwargs.get("action_embed_dim", 32))
    actor_hidden_sizes = tuple(config.algo_kwargs.get("actor_hidden_sizes", (256, 256)))
    critic_hidden_sizes = tuple(config.algo_kwargs.get("critic_hidden_sizes", (256, 256)))
    reward_hidden_sizes = tuple(config.algo_kwargs.get("reward_hidden_sizes", (256, 256)))

    world_model_learning_rate = float(config.algo_kwargs.get("world_model_learning_rate", 1e-3))
    actor_learning_rate = float(config.algo_kwargs.get("actor_learning_rate", 3e-4))
    critic_learning_rate = float(config.algo_kwargs.get("critic_learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    entropy_coef = resolve_entropy_coefficient(config, step=0, coefficient_key="entropy_coef", default=1e-3)
    eval_interval = resolve_eval_interval(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_shape, action_dim = _infer_spaces(envs)
        if config.algo == "po_dreamer":
            model_cls = PODreamerModel
        elif config.algo == "mow":
            model_cls = MoWModel
        elif config.algo == "eadream":
            model_cls = EADreamModel
        else:
            model_cls = DreamerModel
        model_kwargs = {
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "features_dim": features_dim,
            "action_embed_dim": action_embed_dim,
            "actor_hidden_sizes": actor_hidden_sizes,
            "critic_hidden_sizes": critic_hidden_sizes,
            "reward_hidden_sizes": reward_hidden_sizes,
        }
        if model_cls is PODreamerModel:
            model_kwargs["memory_dim"] = int(config.algo_kwargs.get("memory_dim", 64))
            model_kwargs["memory_hidden_size"] = int(config.algo_kwargs.get("memory_hidden_size", 128))
            model_kwargs["memory_mix"] = float(config.algo_kwargs.get("memory_mix", 0.35))
        if model_cls is MoWModel:
            model_kwargs["num_experts"] = int(config.algo_kwargs.get("num_experts", 4))
            model_kwargs["gating_hidden_size"] = int(config.algo_kwargs.get("gating_hidden_size", 128))
        if model_cls is EADreamModel:
            model_kwargs["event_hidden_sizes"] = tuple(config.algo_kwargs.get("event_hidden_sizes", (128,)))
            model_kwargs["event_scale"] = float(config.algo_kwargs.get("event_scale", 1.0))
        model = model_cls(**model_kwargs).to(device)
        if config.algo == "dreamerv3":
            algorithm_cls = DreamerV3
        elif config.algo == "diamond":
            algorithm_cls = Diamond
        elif config.algo == "horizon_imagination":
            algorithm_cls = HorizonImagination
        elif config.algo == "po_dreamer":
            algorithm_cls = PODreamer
        elif config.algo == "twisted":
            algorithm_cls = Twisted
        elif config.algo == "eadream":
            algorithm_cls = EADream
        elif config.algo == "mow":
            algorithm_cls = MoW
        else:
            algorithm_cls = Dreamer
        algorithm_kwargs = {
            "model": model,
            "world_model_learning_rate": world_model_learning_rate,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
        }
        if algorithm_cls is DreamerV3:
            algorithm_kwargs["unimix_ratio"] = float(config.algo_kwargs.get("unimix_ratio", 0.01))
        if algorithm_cls in {Diamond, HorizonImagination}:
            algorithm_kwargs["denoising_loss_coef"] = float(config.algo_kwargs.get("denoising_loss_coef", 0.5))
            algorithm_kwargs["noise_scale"] = float(config.algo_kwargs.get("noise_scale", 0.15))
            algorithm_kwargs["denoiser_hidden_channels"] = int(config.algo_kwargs.get("denoiser_hidden_channels", 64))
        if algorithm_cls is HorizonImagination:
            algorithm_kwargs["stabilization_coef"] = float(config.algo_kwargs.get("stabilization_coef", 0.25))
            algorithm_kwargs["schedule_bias"] = float(config.algo_kwargs.get("schedule_bias", 0.5))
            algorithm_kwargs["subframe_budget_ratio"] = float(config.algo_kwargs.get("subframe_budget_ratio", 0.5))
        if algorithm_cls is PODreamer:
            algorithm_kwargs["memory_loss_coef"] = float(config.algo_kwargs.get("memory_loss_coef", 0.5))
        if algorithm_cls is Twisted:
            algorithm_kwargs["reuse_loss_coef"] = float(config.algo_kwargs.get("reuse_loss_coef", 0.5))
            algorithm_kwargs["reuse_threshold"] = float(config.algo_kwargs.get("reuse_threshold", 0.03))
            algorithm_kwargs["transport_temperature"] = float(
                config.algo_kwargs.get("transport_temperature", 0.5)
            )
        if algorithm_cls is EADream:
            algorithm_kwargs["event_loss_coef"] = float(config.algo_kwargs.get("event_loss_coef", 0.5))
            algorithm_kwargs["event_threshold"] = float(config.algo_kwargs.get("event_threshold", 0.01))
        algorithm = algorithm_cls(**algorithm_kwargs)

        replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            action_shape=(),
            device=device,
            obs_dtype=torch.uint8,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = int(checkpoint_state.trainer_state.get("update_count", 0)) if checkpoint_state is not None else 0
        latest_world_model_metrics: MetricDict = {}
        latest_actor_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                rollout = algorithm.act(obs_tensor, deterministic=False)
                actions = rollout.actions

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            store_vector_transitions(
                replay_buffer,
                obs=obs,
                actions=actions,
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

            current_entropy_coef = resolve_entropy_coefficient(
                config,
                step=global_step,
                coefficient_key="entropy_coef",
                default=1e-3,
            )
            algorithm.entropy_coef = current_entropy_coef
            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                for _ in range(max(0, world_model_updates)):
                    result = algorithm.update_world_model(replay_buffer.sample(batch_size), global_step=global_step)
                    latest_world_model_metrics = result.metrics
                    update_count += result.num_gradient_steps
                    callback_list.on_update_end(trainer_state, result)

                for _ in range(max(0, actor_critic_updates)):
                    start_obs = replay_buffer.sample(imagination_batch_size)["obs"].to(dtype=torch.float32)
                    result = algorithm.update_actor_critic(
                        start_obs,
                        imagination_horizon=imagination_horizon,
                        global_step=global_step,
                    )
                    latest_actor_metrics = result.metrics
                    update_count += result.num_gradient_steps
                    callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_world_model_metrics,
                **latest_actor_metrics,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
                "entropy_coef": float(current_entropy_coef),
            }

            if should_run_periodic_eval(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                algorithm.set_eval_mode()
                eval_metrics = _evaluate_policy(model, config, device=device, num_episodes=config.eval_episodes)
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
                "update_count": update_count,
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
