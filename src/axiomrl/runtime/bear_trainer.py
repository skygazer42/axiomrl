from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from axiomrl.algorithms.bear import BEAR
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.mlp_bear import MLPBEARModel
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import (
    resolve_effective_total_updates,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from axiomrl.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces
from axiomrl.runtime.resume_state import capture_global_random_state, restore_global_random_state
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.sac_trainer import _evaluate_sac_policy
from axiomrl.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainResult
from axiomrl.runtime.types import MetricDict


def train_bear(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    session = create_training_session(config, algorithm="bear", run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

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

        model = MLPBEARModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = BEAR(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            behavior_kl_weight=float(config.algo_kwargs.get("behavior_kl_weight", 0.5)),
            mmd_sigma=float(config.algo_kwargs.get("mmd_sigma", 20.0)),
            mmd_alpha=float(config.algo_kwargs.get("mmd_alpha", 10.0)),
            num_mmd_action_samples=int(config.algo_kwargs.get("num_mmd_action_samples", 10)),
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            resume_context = checkpoint_state.trainer_state.get("resume_context")
            if isinstance(resume_context, dict):
                random_state = resume_context.get("random_state")
                if isinstance(random_state, dict):
                    restore_global_random_state(random_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        epoch = int(checkpoint_state.trainer_state.get("epoch", global_step)) if checkpoint_state is not None else 0
        update_count = (
            int(checkpoint_state.trainer_state.get("update_count", global_step)) if checkpoint_state is not None else 0
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
                eval_metrics = _evaluate_sac_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
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
                "resume_context": {
                    "random_state": capture_global_random_state(),
                },
            },
            metrics=metrics,
        )
    finally:
        session.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
