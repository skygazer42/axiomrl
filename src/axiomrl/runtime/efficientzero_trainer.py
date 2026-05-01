from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from axiomrl.algorithms.efficientzero import EfficientZero
from axiomrl.algorithms.muzero import MuZeroMCTSConfig
from axiomrl.data.muzero_replay_buffer import MuZeroReplayBuffer
from axiomrl.envs.factory import build_env
from axiomrl.experiment.checkpointing import CheckpointState
from axiomrl.experiment.config import TrainConfig
from axiomrl.models.muzero import MuZeroModel
from axiomrl.runtime.callbacks import Callback
from axiomrl.runtime.controls import (
    resolve_eval_interval,
    resolve_num_simulations,
    resolve_root_exploration_fraction,
    resolve_temperature,
    should_run_evaluation,
)
from axiomrl.runtime.muzero_trainer import (
    _emit_collect_event,
    _evaluate_muzero_policy,
    _infer_spaces,
    _maybe_run_muzero_evaluation,
    _restore_training_state,
)
from axiomrl.runtime.resume_state import capture_env_resume_state, capture_global_random_state
from axiomrl.runtime.run_utils import save_training_checkpoint
from axiomrl.runtime.session import create_training_session
from axiomrl.runtime.trainer import TrainResult
from axiomrl.runtime.types import MetricDict


def train_efficientzero(
    config: TrainConfig,
    *,
    run_suffix: str | None = None,
    checkpoint_state: CheckpointState | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> TrainResult:
    if config.num_envs != 1:
        raise ValueError("EfficientZero MVP currently supports num_envs=1 only")

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
    consistency_loss_weight = float(config.algo_kwargs.get("consistency_loss_weight", 1.0))
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
        model = MuZeroModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            latent_dim=int(config.algo_kwargs.get("latent_dim", 256)),
            action_embed_dim=int(config.algo_kwargs.get("action_embed_dim", 64)),
            dynamics_hidden_sizes=tuple(config.algo_kwargs.get("dynamics_hidden_sizes", (256,))),
            prediction_hidden_sizes=tuple(config.algo_kwargs.get("prediction_hidden_sizes", (256,))),
            normalize_latent=bool(config.algo_kwargs.get("normalize_latent", True)),
        ).to(device)

        algorithm = EfficientZero(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            mcts_config=mcts_config,
            unroll_steps=unroll_steps,
            value_loss_weight=value_loss_weight,
            reward_loss_weight=reward_loss_weight,
            policy_loss_weight=policy_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
            max_grad_norm=max_grad_norm,
        )

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
                batch = {
                    "obs": batch["obs"].to(device=device, dtype=torch.float32),
                    "target_obs": batch["target_obs"].to(device=device, dtype=torch.float32),
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


__all__ = ["train_efficientzero", "_evaluate_muzero_policy"]
