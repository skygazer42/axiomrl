from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.bcq import BCQ
from rl_training.envs.factory import build_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_bcq import MLPBCQModel
from rl_training.runtime.callbacks import Callback, CallbackList, merge_callbacks
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    build_control_callbacks,
    resolve_effective_total_updates,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from rl_training.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.td3_trainer import _scale_actions
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _evaluate_bcq_policy(
    model: MLPBCQModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
    num_action_samples: int,
) -> MetricDict:
    env = build_env(config, 0, evaluation=True)
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"unsupported action space for BCQ evaluation: {type(action_space)!r}")

    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
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
                    normalized_action = model.select_actions(
                        obs_tensor,
                        num_action_samples=num_action_samples,
                        deterministic=True,
                    ).squeeze(0)
                    env_action = _scale_actions(normalized_action, low=low, high=high)
                obs, reward, done, truncated, _ = env.step(env_action.cpu().numpy())
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_episodes": float(len(returns)),
    }


def train_bcq(
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
    trainer_state = TrainerState(algorithm="bcq", run_dir=run_context.run_dir)

    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    eval_interval = resolve_eval_interval(config)
    effective_total_updates = resolve_effective_total_updates(config)
    max_updates = resolve_max_updates(config)
    max_epochs = resolve_max_epochs(config)
    warmup_steps = int(config.algo_kwargs.get("warmup_steps", 0))
    learning_rate_schedule = config.algo_kwargs.get("learning_rate_schedule")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_space, action_space = _infer_env_spaces(config)
        dataset = _build_offline_dataset(config, action_space=action_space)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])
        latent_dim = int(config.algo_kwargs.get("latent_dim", action_dim * 2))
        num_action_samples = int(config.algo_kwargs.get("num_action_samples", 10))
        perturbation_scale = float(config.algo_kwargs.get("perturbation_scale", 0.05))
        vae_kl_weight = float(config.algo_kwargs.get("vae_kl_weight", 0.5))

        model = MLPBCQModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            perturbation_scale=perturbation_scale,
            num_action_samples=num_action_samples,
        ).to(device)
        algorithm = BCQ(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            num_action_samples=num_action_samples,
            vae_kl_weight=vae_kl_weight,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        epoch = int(checkpoint_state.trainer_state.get("epoch", global_step)) if checkpoint_state is not None else 0
        update_count = (
            int(checkpoint_state.trainer_state.get("update_count", global_step))
            if checkpoint_state is not None
            else 0
        )
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        trainer_state.epoch = epoch
        trainer_state.update_count = update_count
        initial_stop_reason = stop_reason_for_training_limits(
            epoch=epoch,
            update_count=update_count,
            max_epochs=max_epochs,
            max_updates=max_updates,
        )
        if initial_stop_reason is not None:
            trainer_state.request_stop(initial_stop_reason)
        callback_list.on_train_start(trainer_state)
        callback_list.on_collect_end(
            trainer_state,
            CollectResult(
                num_env_steps=len(dataset),
                num_episodes=0,
                metrics={"dataset_size": float(len(dataset))},
                last_obs=None,
            ),
        )

        while global_step < config.total_timesteps and not trainer_state.should_stop:
            lr_scale = resolve_schedule_value(
                learning_rate_schedule,
                step=update_count,
                total_steps=effective_total_updates,
                warmup_steps=warmup_steps,
            )
            current_learning_rate = apply_learning_rate_scale(algorithm, scale=lr_scale)
            result = algorithm.update(dataset.sample(batch_size, device=device), global_step=global_step)
            global_step += 1
            epoch += 1
            update_count += result.num_gradient_steps
            latest_update_metrics = result.metrics
            trainer_state.global_step = global_step
            trainer_state.epoch = epoch
            trainer_state.update_count = update_count
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "global_step": float(global_step),
                "epoch": float(epoch),
                "update_count": float(update_count),
                "gradient_steps": float(update_count),
                "dataset_size": float(len(dataset)),
                "lr_scale": float(lr_scale),
                "learning_rate": float(current_learning_rate),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                eval_metrics = _evaluate_bcq_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                    num_action_samples=num_action_samples,
                )
                metrics = {**metrics, **eval_metrics}
                logger.log_metrics(metrics, step=global_step)
                callback_list.on_eval_end(trainer_state, metrics)
                if trainer_state.should_stop:
                    break

            stop_reason = stop_reason_for_training_limits(
                epoch=epoch,
                update_count=update_count,
                max_epochs=max_epochs,
                max_updates=max_updates,
            )
            if stop_reason is not None:
                trainer_state.request_stop(stop_reason)
                break

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
            trainer_state={
                "global_step": global_step,
                "epoch": epoch,
                "update_count": update_count,
                "should_stop": trainer_state.should_stop,
                "stop_reason": trainer_state.stop_reason,
            },
            metrics=metrics,
        )
    finally:
        run_artifacts.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
