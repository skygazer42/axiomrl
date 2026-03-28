from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.gumbel_muzero import GumbelMuZero
from rl_training.algorithms.muzero import MuZero, MuZeroMCTSConfig
from rl_training.algorithms.scalezero import ScaleZero
from rl_training.data.muzero_replay_buffer import MuZeroReplayBuffer
from rl_training.envs.factory import build_env
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.muzero import MuZeroModel
from rl_training.models.scalezero import ScaleZeroModel
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.controls import (
    resolve_eval_interval,
    resolve_num_simulations,
    resolve_root_exploration_fraction,
    resolve_temperature,
    should_run_evaluation,
)
from rl_training.runtime.evaluation_support import evaluate_discrete_episodes
from rl_training.runtime.resume_state import (
    capture_env_resume_state,
    capture_global_random_state,
    restore_env_resume_state,
    restore_global_random_state,
)
from rl_training.runtime.run_utils import save_training_checkpoint
from rl_training.runtime.session import create_training_session
from rl_training.runtime.trainer import TrainerState, TrainResult
from rl_training.runtime.types import MetricDict


def _infer_spaces(env: gym.Env) -> tuple[tuple[int, ...], int]:
    obs_space = env.observation_space
    action_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for MuZero trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for MuZero trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 3:
        raise ValueError(f"MuZero MVP expects channel-first image observations, got shape={obs_space.shape!r}")

    return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)


def _evaluate_muzero_policy(
    algorithm: MuZero,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    algorithm.set_eval_mode()

    def action_fn(obs_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            action = algorithm.act(obs_tensor, deterministic=True).actions.squeeze(0)
        return int(action.item())

    return evaluate_discrete_episodes(
        config,
        device=device,
        num_episodes=num_episodes,
        action_fn=action_fn,
    )


def _emit_collect_event(
    callback_list: CallbackList,
    trainer_state: TrainerState,
    *,
    global_step: int,
    done: bool,
    replay_size: int,
    obs: np.ndarray,
) -> None:
    callback_list.on_collect_end(
        trainer_state,
        CollectResult(
            num_env_steps=1,
            num_episodes=int(bool(done)),
            metrics={"global_step": float(global_step), "buffer_size": float(replay_size)},
            last_obs=obs,
        ),
    )


def _maybe_run_muzero_evaluation(
    *,
    should_run_eval: bool,
    algorithm: MuZero,
    config: TrainConfig,
    device: torch.device,
    logger: object,
    callback_list: CallbackList,
    trainer_state: TrainerState,
    metrics: MetricDict,
    global_step: int,
) -> tuple[MetricDict, bool]:
    if not should_run_eval:
        return metrics, False

    eval_metrics = _evaluate_muzero_policy(
        algorithm,
        config,
        device=device,
        num_episodes=config.eval_episodes,
    )
    evaluated_metrics = {**metrics, **eval_metrics}
    logger.log_metrics(evaluated_metrics, step=global_step)
    callback_list.on_eval_end(trainer_state, evaluated_metrics)
    return evaluated_metrics, trainer_state.should_stop


def _restore_training_state(
    *,
    algorithm: MuZero,
    replay_buffer: MuZeroReplayBuffer,
    env: gym.Env,
    checkpoint_state: CheckpointState | None,
) -> tuple[object | None, int, int]:
    if checkpoint_state is None:
        return None, 0, 0
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    if checkpoint_state.buffer_state is not None:
        replay_buffer.load_state_dict(checkpoint_state.buffer_state)
    restored_obs = None
    resume_context = checkpoint_state.trainer_state.get("resume_context")
    if isinstance(resume_context, dict):
        env_resume_state = resume_context.get("env_state")
        if isinstance(env_resume_state, dict):
            restored_obs = restore_env_resume_state(env, env_resume_state)
        random_state = resume_context.get("random_state")
        if isinstance(random_state, dict):
            restore_global_random_state(random_state)
    return (
        restored_obs,
        int(checkpoint_state.trainer_state.get("global_step", 0)),
        int(checkpoint_state.trainer_state.get("update_count", 0)),
    )


def train_muzero(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if config.num_envs != 1:
        raise ValueError("MuZero MVP currently supports num_envs=1 only")

    session = create_training_session(config, algorithm=config.algo, run_suffix=run_suffix, callbacks=callbacks)
    device = session.device
    run_context = session.run_context
    logger = session.logger
    callback_list = session.callback_list
    trainer_state = session.trainer_state

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 50000))
    batch_size = int(config.algo_kwargs.get("batch_size", 32))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    unroll_steps = int(config.algo_kwargs.get("unroll_steps", 5))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-3))
    gamma = float(config.algo_kwargs.get("gamma", 0.997))
    value_loss_weight = float(config.algo_kwargs.get("value_loss_weight", 1.0))
    reward_loss_weight = float(config.algo_kwargs.get("reward_loss_weight", 1.0))
    policy_loss_weight = float(config.algo_kwargs.get("policy_loss_weight", 1.0))
    max_grad_norm = float(config.algo_kwargs.get("max_grad_norm", 10.0))

    mcts_config = MuZeroMCTSConfig(
        num_simulations=int(config.algo_kwargs.get("num_simulations", 25)),
        pb_c_base=float(config.algo_kwargs.get("pb_c_base", 19652.0)),
        pb_c_init=float(config.algo_kwargs.get("pb_c_init", 1.25)),
        root_dirichlet_alpha=float(config.algo_kwargs.get("root_dirichlet_alpha", 0.3)),
        root_exploration_fraction=float(config.algo_kwargs.get("root_exploration_fraction", 0.25)),
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = build_env(config, 0)
    checkpoint_path: Path | None = None

    try:
        obs_shape, action_dim = _infer_spaces(env)
        model_cls = ScaleZeroModel if config.algo == "scalezero" else MuZeroModel
        model_kwargs = {
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "latent_dim": int(config.algo_kwargs.get("latent_dim", 256)),
            "action_embed_dim": int(config.algo_kwargs.get("action_embed_dim", 64)),
            "dynamics_hidden_sizes": tuple(config.algo_kwargs.get("dynamics_hidden_sizes", (256,))),
            "prediction_hidden_sizes": tuple(config.algo_kwargs.get("prediction_hidden_sizes", (256,))),
            "normalize_latent": bool(config.algo_kwargs.get("normalize_latent", True)),
        }
        if model_cls is ScaleZeroModel:
            model_kwargs["num_experts"] = int(config.algo_kwargs.get("num_experts", 4))
            model_kwargs["gating_hidden_size"] = int(config.algo_kwargs.get("gating_hidden_size", 128))
        model = model_cls(**model_kwargs).to(device)

        if config.algo == "gumbel_muzero":
            algorithm_cls = GumbelMuZero
        elif config.algo == "scalezero":
            algorithm_cls = ScaleZero
        else:
            algorithm_cls = MuZero
        algorithm_kwargs = {
            "model": model,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "mcts_config": mcts_config,
            "unroll_steps": unroll_steps,
            "value_loss_weight": value_loss_weight,
            "reward_loss_weight": reward_loss_weight,
            "policy_loss_weight": policy_loss_weight,
            "max_grad_norm": max_grad_norm,
        }
        if algorithm_cls is GumbelMuZero:
            algorithm_kwargs["gumbel_scale"] = float(config.algo_kwargs.get("gumbel_scale", 1.0))
        algorithm = algorithm_cls(**algorithm_kwargs)

        replay_buffer = MuZeroReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
            action_dim=action_dim,
            device="cpu",
            obs_dtype=torch.uint8,
        )

        obs, _ = env.reset(seed=config.seed)
        restored_obs, global_step, update_count = _restore_training_state(
            algorithm=algorithm,
            replay_buffer=replay_buffer,
            env=env,
            checkpoint_state=checkpoint_state,
        )
        if restored_obs is not None:
            obs = restored_obs
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        trainer_state.update_count = update_count
        callback_list.on_train_start(trainer_state)

        eval_interval = resolve_eval_interval(config)

        while global_step < config.total_timesteps:
            current_temperature = resolve_temperature(config, step=global_step, default=1.0)
            current_root_exploration_fraction = resolve_root_exploration_fraction(
                config,
                step=global_step,
                default=mcts_config.root_exploration_fraction,
            )
            current_num_simulations = resolve_num_simulations(
                config,
                step=global_step,
                default=mcts_config.num_simulations,
            )
            action, policy, root_value = algorithm.plan(
                obs,
                temperature=current_temperature,
                add_root_noise=True,
                deterministic=False,
                root_exploration_fraction=current_root_exploration_fraction,
                num_simulations=current_num_simulations,
            )
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = bool(terminated or truncated)

            replay_buffer.add(
                obs=obs,
                action=int(action),
                reward=float(reward),
                done=done,
                policy=policy,
                next_obs=next_obs,
                step=global_step,
            )

            obs = next_obs
            if done:
                obs, _ = env.reset(seed=config.seed + global_step + 1)

            global_step += 1
            trainer_state.global_step = global_step
            _emit_collect_event(
                callback_list,
                trainer_state,
                global_step=global_step,
                done=done,
                replay_size=len(replay_buffer),
                obs=np.asarray(obs),
            )

            if len(replay_buffer) >= max(learning_starts, batch_size) and global_step % train_frequency == 0:
                batch = replay_buffer.sample(batch_size, unroll_steps=unroll_steps)
                # Move batch tensors to the model device as float32 where appropriate.
                batch = {
                    "obs": batch["obs"].to(device=device, dtype=torch.float32),
                    "bootstrap_obs": batch["bootstrap_obs"].to(device=device, dtype=torch.float32),
                    "actions": batch["actions"].to(device=device),
                    "rewards": batch["rewards"].to(device=device),
                    "dones": batch["dones"].to(device=device),
                    "target_policies": batch["target_policies"].to(device=device),
                }
                update_result = algorithm.update(batch, global_step=global_step)
                latest_update_metrics = update_result.metrics
                update_count += update_result.num_gradient_steps
                trainer_state.update_count = update_count
                callback_list.on_update_end(trainer_state, update_result)

            metrics: MetricDict = {
                **latest_update_metrics,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
                "root_value": float(root_value),
                "temperature": float(current_temperature),
                "root_exploration_fraction": float(current_root_exploration_fraction),
                "num_simulations": float(current_num_simulations),
            }
            if isinstance(algorithm, GumbelMuZero):
                metrics["gumbel_scale"] = float(algorithm.gumbel_scale)
            metrics, should_stop = _maybe_run_muzero_evaluation(
                should_run_eval=should_run_evaluation(
                    global_step=global_step,
                    total_timesteps=config.total_timesteps,
                    eval_interval=eval_interval,
                ),
                algorithm=algorithm,
                config=config,
                device=device,
                logger=logger,
                callback_list=callback_list,
                trainer_state=trainer_state,
                metrics=metrics,
                global_step=global_step,
            )
            if should_stop:
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
                "resume_context": {
                    "env_state": capture_env_resume_state(env),
                    "random_state": capture_global_random_state(),
                },
            },
            metrics=metrics,
        )
    finally:
        env.close()
        session.close()

    result = TrainResult(run_dir=run_context.run_dir, checkpoint_path=checkpoint_path, metrics=metrics)
    callback_list.on_train_end(trainer_state, result)
    return result
