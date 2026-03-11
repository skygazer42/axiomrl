from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rl_training.algorithms.td3_bc import TD3BC
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_td3 import MLPTD3Model
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.iql_trainer import _build_offline_dataset, _infer_env_spaces
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.td3_trainer import _evaluate_td3_policy
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def train_td3_bc(
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
    callback_list = CallbackList(callbacks)
    trainer_state = TrainerState(algorithm="td3_bc", run_dir=run_context.run_dir)

    batch_size = int(config.algo_kwargs.get("batch_size", 256))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 3e-4))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    tau = float(config.algo_kwargs.get("tau", 0.005))
    policy_noise = float(config.algo_kwargs.get("policy_noise", 0.2))
    noise_clip = float(config.algo_kwargs.get("noise_clip", 0.5))
    policy_delay = int(config.algo_kwargs.get("policy_delay", 2))
    bc_alpha = float(config.algo_kwargs.get("bc_alpha", 2.5))

    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        obs_space, action_space = _infer_env_spaces(config)
        dataset = _build_offline_dataset(config)
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])

        model = MLPTD3Model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm = TD3BC(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            bc_alpha=bc_alpha,
        )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)

        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
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

        while global_step < config.total_timesteps:
            result = algorithm.update(dataset.sample(batch_size, device=device), global_step=global_step)
            global_step += 1
            update_count += result.num_gradient_steps
            latest_update_metrics = result.metrics
            trainer_state.global_step = global_step
            callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "global_step": float(global_step),
                "gradient_steps": float(update_count),
                "dataset_size": float(len(dataset)),
            }

        eval_metrics = _evaluate_td3_policy(
            model,
            config,
            device=device,
            num_episodes=config.eval_episodes,
        )
        metrics = {**metrics, **eval_metrics}
        logger.log_metrics(metrics, step=global_step)
        callback_list.on_eval_end(trainer_state, eval_metrics)

        checkpoint_path = save_training_checkpoint(
            run_context=run_context,
            config=config,
            algorithm_state=algorithm.state_dict(),
            buffer_state=None,
            trainer_state={"global_step": global_step},
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
