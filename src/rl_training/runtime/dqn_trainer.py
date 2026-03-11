from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.c51_dqn import C51DQN
from rl_training.algorithms.dqn import DQN, DoubleDQN, DuelingDQN, PrioritizedDQN, RainbowDQN
from rl_training.algorithms.iqn import IQN
from rl_training.algorithms.qr_dqn import QRDQN
from rl_training.data.n_step import NStepAccumulator
from rl_training.data.prioritized_replay_buffer import PrioritizedReplayBuffer
from rl_training.data.replay_buffer import ReplayBuffer
from rl_training.envs.factory import make_vector_env
from rl_training.experiment.checkpointing import CheckpointState, save_checkpoint
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_c51_q_network import MLPC51QNetwork
from rl_training.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from rl_training.models.mlp_dueling_q_network import MLPDuelingQNetwork
from rl_training.models.mlp_iqn_network import MLPIQNetwork
from rl_training.models.mlp_noisy_q_network import MLPNoisyQNetwork
from rl_training.models.mlp_q_network import MLPQNetwork
from rl_training.models.mlp_qr_q_network import MLPQRQNetwork
from rl_training.runtime.callbacks import Callback, CallbackList
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.run_utils import create_training_run, resolve_device, save_training_checkpoint
from rl_training.runtime.trainer import TrainResult, TrainerState
from rl_training.runtime.types import MetricDict


def _infer_spaces(envs: gym.vector.SyncVectorEnv) -> tuple[int, int]:
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise TypeError(f"unsupported observation space for DQN trainer: {type(obs_space)!r}")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"unsupported action space for DQN trainer: {type(action_space)!r}")
    if obs_space.shape is None or len(obs_space.shape) != 1:
        raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")

    return int(obs_space.shape[0]), int(action_space.n)


def _epsilon_at_step(
    step: int,
    *,
    total_timesteps: int,
    epsilon_start: float,
    epsilon_end: float,
    exploration_fraction: float,
) -> float:
    decay_steps = max(1, int(total_timesteps * exploration_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(epsilon_start + progress * (epsilon_end - epsilon_start))


def _beta_at_step(
    step: int,
    *,
    total_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_fraction: float,
) -> float:
    if beta_fraction <= 0:
        return float(beta_end)
    decay_steps = max(1, int(total_timesteps * beta_fraction))
    progress = min(step / decay_steps, 1.0)
    return float(beta_start + progress * (beta_end - beta_start))


def _build_q_network(
    config: TrainConfig,
    *,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> (
    MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork
):
    if config.algo == "dueling_dqn":
        return MLPDuelingQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "noisy_dqn":
        return MLPNoisyQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "rainbow_dqn":
        return MLPDuelingNoisyQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "c51_dqn":
        return MLPC51QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            v_min=float(config.algo_kwargs.get("v_min", 0.0)),
            v_max=float(config.algo_kwargs.get("v_max", 200.0)),
            num_atoms=int(config.algo_kwargs.get("num_atoms", 51)),
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "qr_dqn":
        return MLPQRQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=int(config.algo_kwargs.get("num_quantiles", 51)),
            hidden_sizes=hidden_sizes,
        )
    if config.algo == "iqn":
        return MLPIQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_quantiles=int(config.algo_kwargs.get("num_quantiles", 32)),
            hidden_sizes=hidden_sizes,
            embedding_dim=int(config.algo_kwargs.get("embedding_dim", 64)),
        )
    return MLPQNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )


def _build_algorithm(
    config: TrainConfig,
    *,
    q_network: MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork,
    learning_rate: float,
    gamma: float,
    target_update_interval: int,
) -> DQN | C51DQN | QRDQN | IQN:
    if config.algo == "c51_dqn":
        if not isinstance(q_network, MLPC51QNetwork):
            raise TypeError(f"expected MLPC51QNetwork for c51_dqn, got {type(q_network)!r}")
        return C51DQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            v_min=float(config.algo_kwargs.get("v_min", q_network.v_min)),
            v_max=float(config.algo_kwargs.get("v_max", q_network.v_max)),
            num_atoms=int(config.algo_kwargs.get("num_atoms", q_network.num_atoms)),
        )
    if config.algo == "qr_dqn":
        if not isinstance(q_network, MLPQRQNetwork):
            raise TypeError(f"expected MLPQRQNetwork for qr_dqn, got {type(q_network)!r}")
        num_quantiles = int(config.algo_kwargs.get("num_quantiles", q_network.num_quantiles))
        return QRDQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            num_quantiles=num_quantiles,
            kappa=float(config.algo_kwargs.get("kappa", 1.0)),
        )
    if config.algo == "iqn":
        if not isinstance(q_network, MLPIQNetwork):
            raise TypeError(f"expected MLPIQNetwork for iqn, got {type(q_network)!r}")
        num_quantiles = int(config.algo_kwargs.get("num_quantiles", q_network.num_quantiles))
        return IQN(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            num_quantiles=num_quantiles,
            kappa=float(config.algo_kwargs.get("kappa", 1.0)),
        )
    if config.algo == "double_dqn":
        algorithm_cls = DoubleDQN
    elif config.algo == "dueling_dqn":
        algorithm_cls = DuelingDQN
    elif config.algo == "prioritized_dqn":
        algorithm_cls = PrioritizedDQN
    elif config.algo == "rainbow_dqn":
        algorithm_cls = RainbowDQN
    else:
        algorithm_cls = DQN

    return algorithm_cls(
        q_network=q_network,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_interval=target_update_interval,
    )


def _evaluate_q_policy(
    q_network: MLPQNetwork
    | MLPDuelingQNetwork
    | MLPNoisyQNetwork
    | MLPDuelingNoisyQNetwork
    | MLPC51QNetwork
    | MLPQRQNetwork
    | MLPIQNetwork,
    config: TrainConfig,
    *,
    device: torch.device,
    num_episodes: int,
) -> MetricDict:
    env = gym.make(config.env_id, **config.env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
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
                    action = q_network.act(obs_tensor, epsilon=0.0).squeeze(0)
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


def train_dqn(
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
    trainer_state = TrainerState(algorithm=config.algo, run_dir=run_context.run_dir)

    buffer_capacity = int(config.algo_kwargs.get("buffer_capacity", 10000))
    batch_size = int(config.algo_kwargs.get("batch_size", 64))
    learning_starts = int(config.algo_kwargs.get("learning_starts", 1000))
    train_frequency = int(config.algo_kwargs.get("train_frequency", 1))
    target_update_interval = int(config.algo_kwargs.get("target_update_interval", 250))
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    learning_rate = float(config.algo_kwargs.get("learning_rate", 1e-3))
    gamma = float(config.algo_kwargs.get("gamma", 0.99))
    n_step = int(config.algo_kwargs.get("n_step", 1))
    epsilon_start = float(config.algo_kwargs.get("epsilon_start", 1.0))
    epsilon_end = float(config.algo_kwargs.get("epsilon_end", 0.05))
    exploration_fraction = float(config.algo_kwargs.get("exploration_fraction", 0.3))
    prioritized_alpha = float(config.algo_kwargs.get("prioritized_alpha", 0.6))
    prioritized_beta_start = float(config.algo_kwargs.get("prioritized_beta_start", 0.4))
    prioritized_beta_end = float(config.algo_kwargs.get("prioritized_beta_end", 1.0))
    prioritized_beta_fraction = float(config.algo_kwargs.get("prioritized_beta_fraction", 1.0))
    prioritized_eps = float(config.algo_kwargs.get("prioritized_eps", 1e-6))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vector_env(config)
    checkpoint_path: Path | None = None
    metrics: MetricDict = {}

    try:
        if n_step <= 0:
            raise ValueError(f"n_step must be > 0, got {n_step}")

        obs_dim, action_dim = _infer_spaces(envs)
        q_network = _build_q_network(
            config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        algorithm_gamma = gamma
        if config.algo in {"n_step_dqn", "rainbow_dqn"} and n_step > 1:
            algorithm_gamma = gamma**n_step

        algorithm = _build_algorithm(
            config,
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=algorithm_gamma,
            target_update_interval=target_update_interval,
        )
        if config.algo in {"prioritized_dqn", "rainbow_dqn"}:
            replay_buffer: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                obs_shape=(obs_dim,),
                action_shape=(),
                alpha=prioritized_alpha,
                priority_eps=prioritized_eps,
                device=device,
            )
        else:
            replay_buffer = ReplayBuffer(
                capacity=buffer_capacity,
                obs_shape=(obs_dim,),
                action_shape=(),
                device=device,
            )
        if checkpoint_state is not None:
            algorithm.load_state_dict(checkpoint_state.algorithm_state)
            if checkpoint_state.buffer_state is not None:
                replay_buffer.load_state_dict(checkpoint_state.buffer_state)

        n_step_accumulator: NStepAccumulator | None = None
        if config.algo in {"n_step_dqn", "rainbow_dqn"} and n_step > 1:
            n_step_accumulator = NStepAccumulator(num_envs=config.num_envs, n_step=n_step, gamma=gamma)

        obs, _ = envs.reset(seed=config.seed)
        global_step = int(checkpoint_state.trainer_state.get("global_step", 0)) if checkpoint_state is not None else 0
        update_count = 0
        latest_update_metrics: MetricDict = {}
        trainer_state.global_step = global_step
        callback_list.on_train_start(trainer_state)

        while global_step < config.total_timesteps:
            epsilon = _epsilon_at_step(
                global_step,
                total_timesteps=config.total_timesteps,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                exploration_fraction=exploration_fraction,
            )

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions = q_network.act(obs_tensor, epsilon=epsilon)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            for env_index in range(config.num_envs):
                action = int(actions[env_index].item())
                reward = float(rewards[env_index])
                done = bool(dones[env_index])
                if n_step_accumulator is None:
                    replay_buffer.add(
                        obs=obs[env_index],
                        actions=action,
                        rewards=reward,
                        next_obs=next_obs[env_index],
                        dones=float(done),
                    )
                else:
                    for transition in n_step_accumulator.add(
                        env_index,
                        obs[env_index],
                        action,
                        reward,
                        next_obs[env_index],
                        done,
                    ):
                        replay_buffer.add(**transition)

            obs = next_obs
            global_step += config.num_envs
            trainer_state.global_step = global_step
            callback_list.on_collect_end(
                trainer_state,
                CollectResult(
                    num_env_steps=config.num_envs,
                    num_episodes=int(np.sum(dones)),
                    metrics={"global_step": float(global_step), "buffer_size": float(len(replay_buffer))},
                    last_obs=obs,
                ),
            )

            if len(replay_buffer) >= max(batch_size, learning_starts) and global_step % train_frequency == 0:
                if config.algo in {"prioritized_dqn", "rainbow_dqn"}:
                    beta = _beta_at_step(
                        global_step,
                        total_timesteps=config.total_timesteps,
                        beta_start=prioritized_beta_start,
                        beta_end=prioritized_beta_end,
                        beta_fraction=prioritized_beta_fraction,
                    )
                    batch = replay_buffer.sample(batch_size, beta=beta)  # type: ignore[arg-type]
                    result = algorithm.update(batch, global_step=global_step)
                    if algorithm.last_td_errors is not None:
                        replay_buffer.update_priorities(batch["indices"], algorithm.last_td_errors)  # type: ignore[union-attr]
                else:
                    result = algorithm.update(replay_buffer.sample(batch_size), global_step=global_step)  # type: ignore[arg-type]
                latest_update_metrics = result.metrics
                update_count += result.num_gradient_steps
                callback_list.on_update_end(trainer_state, result)

            metrics = {
                **latest_update_metrics,
                "epsilon": epsilon,
                "global_step": float(global_step),
                "buffer_size": float(len(replay_buffer)),
                "gradient_steps": float(update_count),
            }
            if config.algo in {"prioritized_dqn", "rainbow_dqn"}:
                metrics["beta"] = float(
                    _beta_at_step(
                        global_step,
                        total_timesteps=config.total_timesteps,
                        beta_start=prioritized_beta_start,
                        beta_end=prioritized_beta_end,
                        beta_fraction=prioritized_beta_fraction,
                    )
                )

        algorithm.set_eval_mode()
        eval_metrics = _evaluate_q_policy(
            q_network,
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
            buffer_state=replay_buffer.state_dict(),
            trainer_state={"global_step": global_step},
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
