from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.decision_transformer import DecisionTransformer
from rl_training.data.trajectory_windows import TrajectoryWindowDataset
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.decision_transformer import DecisionTransformerModel
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    resolve_effective_total_updates,
    resolve_eval_interval,
    resolve_max_epochs,
    resolve_max_updates,
    should_run_evaluation,
    stop_reason_for_training_limits,
)
from rl_training.runtime.evaluation_support import evaluate_continuous_episodes
from rl_training.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces, _scale_actions
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.schedules import apply_learning_rate_scale, resolve_schedule_value
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


def _build_autoregressive_window(
    obs_history: list[np.ndarray],
    action_history: list[np.ndarray],
    return_history: list[float],
    *,
    context_length: int,
    action_dim: int,
    max_timestep: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sequence_length = min(len(obs_history), int(context_length))
    obs_shape = tuple(int(dim) for dim in obs_history[0].shape)

    obs = torch.zeros((1, context_length, *obs_shape), dtype=torch.float32, device=device)
    actions = torch.zeros((1, context_length, action_dim), dtype=torch.float32, device=device)
    returns_to_go = torch.zeros((1, context_length), dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, context_length), dtype=torch.int64, device=device)
    mask = torch.zeros((1, context_length), dtype=torch.float32, device=device)

    aligned_actions = [*action_history, np.zeros((action_dim,), dtype=np.float32)]
    start_index = len(obs_history) - sequence_length
    dest_start = context_length - sequence_length

    obs_slice = np.stack(obs_history[start_index:], axis=0).astype(np.float32)
    action_slice = np.stack(aligned_actions[start_index:], axis=0).astype(np.float32)
    return_slice = np.asarray(return_history[start_index:], dtype=np.float32)
    timestep_slice = np.arange(start_index, len(obs_history), dtype=np.int64)

    obs[:, dest_start:] = torch.as_tensor(obs_slice, dtype=torch.float32, device=device)
    actions[:, dest_start:] = torch.as_tensor(action_slice, dtype=torch.float32, device=device)
    returns_to_go[:, dest_start:] = torch.as_tensor(return_slice, dtype=torch.float32, device=device)
    timesteps[:, dest_start:] = torch.as_tensor(
        np.clip(timestep_slice, 0, int(max_timestep)),
        dtype=torch.int64,
        device=device,
    )
    mask[:, dest_start:] = 1.0

    return {
        "obs": obs,
        "actions": actions,
        "returns_to_go": returns_to_go,
        "timesteps": timesteps,
        "mask": mask,
    }


def _evaluate_decision_transformer_policy(
    model: DecisionTransformerModel,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
    context_length: int,
    target_return: float,
    max_timestep: int,
    gamma: float,
) -> MetricDict:
    class _ActionFn:
        def __init__(self) -> None:
            self.low: torch.Tensor | None = None
            self.high: torch.Tensor | None = None
            self.action_dim: int | None = None
            self.obs_history: list[np.ndarray] = []
            self.action_history: list[np.ndarray] = []
            self.return_history: list[float] = []
            self.pending_action: np.ndarray | None = None

        def bind_env(self, env: gym.Env) -> None:
            action_space = env.action_space
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError(
                    f"unsupported action space for Decision Transformer evaluation: {type(action_space)!r}"
                )
            self.low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
            self.high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
            self.action_dim = int(action_space.shape[0])

        def reset(self) -> None:
            self.obs_history = []
            self.action_history = []
            self.return_history = [float(target_return)]
            self.pending_action = None

        def __call__(self, obs_tensor: torch.Tensor) -> np.ndarray:
            if self.low is None or self.high is None or self.action_dim is None:
                raise RuntimeError("action bounds must be bound before evaluation")
            obs = obs_tensor.detach().cpu().numpy().astype(np.float32)
            if not self.obs_history:
                self.obs_history = [obs]

            autoregressive_batch = _build_autoregressive_window(
                self.obs_history,
                self.action_history,
                self.return_history,
                context_length=context_length,
                action_dim=self.action_dim,
                max_timestep=max_timestep,
                device=device,
            )
            with torch.no_grad():
                normalized_action = torch.nan_to_num(
                    model.predict_last_action(**autoregressive_batch).squeeze(0),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                env_action = _scale_actions(normalized_action, low=self.low, high=self.high)
            self.pending_action = normalized_action.detach().cpu().numpy().astype(np.float32)
            return env_action.cpu().numpy()

        def after_step(self, next_obs, reward: float, done: bool, truncated: bool, info) -> None:  # type: ignore[no-untyped-def]
            del info
            if self.pending_action is not None:
                self.action_history.append(self.pending_action)
                self.pending_action = None
            if done or truncated:
                return
            self.obs_history.append(np.asarray(next_obs, dtype=np.float32))
            gamma_value = float(gamma)
            next_return = 0.0 if gamma_value <= 1e-8 else (self.return_history[-1] - float(reward)) / gamma_value
            self.return_history.append(float(next_return))

    return evaluate_continuous_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=_ActionFn(),
    )


def train_decision_transformer(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if config.algo_kwargs.get("dataset_mix") not in (None, False):
        raise ValueError("decision_transformer does not support dataset_mix because it destroys trajectory order")

    session = create_training_session(
        config,
        algorithm="decision_transformer",
        run_suffix=run_suffix,
        callbacks=callbacks,
    )
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    batch_size = int(config.algo_kwargs.get("batch_size", 64))
    context_length = int(config.algo_kwargs.get("context_length", 20))
    hidden_size = int(config.algo_kwargs.get("hidden_size", 128))
    num_layers = int(config.algo_kwargs.get("num_layers", 3))
    num_heads = int(config.algo_kwargs.get("num_heads", 4))
    dropout = float(config.algo_kwargs.get("dropout", 0.1))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    target_return = float(config.algo_kwargs.get("target_return", 0.0))
    max_timestep = int(config.algo_kwargs.get("max_timestep", 1024))
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
        transition_dataset = _build_offline_dataset(config, action_space=action_space).with_discounted_returns_to_go(
            gamma=gamma
        )
        window_dataset = TrajectoryWindowDataset.from_transition_dataset(
            transition_dataset,
            context_length=context_length,
        )
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])

        model = DecisionTransformerModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_length=context_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_timestep=max_timestep,
            dropout=dropout,
        ).to(device)
        algorithm = DecisionTransformer(
            model=model,
            learning_rate=learning_rate,
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
                num_env_steps=len(transition_dataset),
                num_episodes=0,
                metrics={
                    "dataset_size": float(len(transition_dataset)),
                    "window_count": float(len(window_dataset)),
                },
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
            result = algorithm.update(window_dataset.sample(batch_size, device=device), global_step=global_step)
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
                "dataset_size": float(len(transition_dataset)),
                "window_count": float(len(window_dataset)),
                "lr_scale": float(lr_scale),
                "learning_rate": float(current_learning_rate),
            }
            if should_run_evaluation(
                global_step=global_step,
                total_timesteps=config.total_timesteps,
                eval_interval=eval_interval,
            ):
                eval_metrics = _evaluate_decision_transformer_policy(
                    model,
                    config,
                    device=device,
                    num_episodes=config.eval_episodes,
                    context_length=context_length,
                    target_return=target_return,
                    max_timestep=max_timestep,
                    gamma=gamma,
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
        session.close()

    result = TrainResult(
        run_dir=run_context.run_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
    callback_list.on_train_end(trainer_state, result)
    return result
