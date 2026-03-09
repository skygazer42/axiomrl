from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.experiment.checkpointing import CheckpointState
from rl_training.experiment.config import TrainConfig
from rl_training.models.mlp_actor_critic import MLPActorCritic
from rl_training.models.mlp_q_network import MLPQNetwork
from rl_training.models.mlp_sac import MLPSACModel
from rl_training.models.mlp_td3 import MLPTD3Model
from rl_training.runtime.a2c_trainer import _evaluate_policy as _evaluate_a2c_policy
from rl_training.runtime.a2c_trainer import train_a2c
from rl_training.runtime.dqn_trainer import _evaluate_q_policy, train_dqn
from rl_training.runtime.ppo_trainer import _evaluate_policy, train_ppo
from rl_training.runtime.sac_trainer import _evaluate_sac_policy, train_sac
from rl_training.runtime.td3_trainer import _evaluate_td3_policy, train_td3
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.types import MetricDict


TrainFn = Callable[..., TrainResult]
EvaluateFn = Callable[[TrainConfig, CheckpointState, torch.device, int], MetricDict]
PredictFn = Callable[[TrainConfig, CheckpointState, torch.device, object, bool], int | np.ndarray]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    name: str
    train_fn: TrainFn
    evaluate_fn: EvaluateFn
    predict_fn: PredictFn


def _infer_discrete_env_spaces(config: TrainConfig) -> tuple[int, int]:
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id, **config.env_kwargs)])
    try:
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
        return int(obs_space.shape[0]), int(action_space.n)
    finally:
        envs.close()


def _infer_continuous_env_spaces(config: TrainConfig) -> tuple[int, int]:
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(config.env_id, **config.env_kwargs)])
    try:
        obs_space = envs.single_observation_space
        action_space = envs.single_action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError(f"expected flat 1D observations, got shape={obs_space.shape!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return int(obs_space.shape[0]), int(action_space.shape[0])
    finally:
        envs.close()


def _continuous_action_bounds(config: TrainConfig, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    env = gym.make(config.env_id, **config.env_kwargs)
    try:
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        return low, high
    finally:
        env.close()


def _scale_continuous_actions(
    normalized_actions: torch.Tensor,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    scaled = low + 0.5 * (normalized_actions + 1.0) * (high - low)
    return torch.max(torch.min(scaled, high), low)


def _prepare_observation(obs: object, *, device: torch.device) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor


def _format_action_output(actions: torch.Tensor, *, discrete: bool) -> int | np.ndarray:
    action_tensor = actions.detach().cpu()
    if action_tensor.ndim > 1 and action_tensor.shape[0] == 1:
        action_tensor = action_tensor.squeeze(0)
    if discrete:
        if action_tensor.ndim == 0:
            return int(action_tensor.item())
        if action_tensor.numel() == 1:
            return int(action_tensor.reshape(-1)[0].item())
        return action_tensor.numpy()
    return action_tensor.numpy()


def _load_a2c_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> A2CAlgorithm:
    obs_dim, action_dim = _infer_discrete_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = A2CAlgorithm(
        policy=MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_ppo_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> PPOAlgorithm:
    obs_dim, action_dim = _infer_discrete_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = PPOAlgorithm(
        policy=MLPActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        clip_coef=float(config.algo_kwargs.get("clip_coef", 0.2)),
        ent_coef=float(config.algo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(config.algo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(config.algo_kwargs.get("max_grad_norm", 0.5)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_dqn_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> DQNAlgorithm:
    obs_dim, action_dim = _infer_discrete_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (64, 64)))
    algorithm = DQNAlgorithm(
        q_network=MLPQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 1e-3)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        target_update_interval=int(config.algo_kwargs.get("target_update_interval", 250)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_sac_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> SACAlgorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = SACAlgorithm(
        model=MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        alpha=float(config.algo_kwargs.get("alpha", 0.2)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _load_td3_algorithm(config: TrainConfig, checkpoint_state: CheckpointState, *, device: torch.device) -> TD3Algorithm:
    obs_dim, action_dim = _infer_continuous_env_spaces(config)
    hidden_sizes = tuple(config.algo_kwargs.get("hidden_sizes", (256, 256)))
    algorithm = TD3Algorithm(
        model=MLPTD3Model(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device),
        learning_rate=float(config.algo_kwargs.get("learning_rate", 3e-4)),
        gamma=float(config.algo_kwargs.get("gamma", 0.99)),
        tau=float(config.algo_kwargs.get("tau", 0.005)),
        policy_noise=float(config.algo_kwargs.get("policy_noise", 0.2)),
        noise_clip=float(config.algo_kwargs.get("noise_clip", 0.5)),
        policy_delay=int(config.algo_kwargs.get("policy_delay", 2)),
    )
    algorithm.load_state_dict(checkpoint_state.algorithm_state)
    return algorithm


def _evaluate_a2c(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_a2c_algorithm(config, checkpoint_state, device=device)
    return _evaluate_a2c_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_ppo(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_ppo_algorithm(config, checkpoint_state, device=device)
    return _evaluate_policy(algorithm.policy, config, device=device, num_episodes=num_episodes)


def _evaluate_dqn(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_dqn_algorithm(config, checkpoint_state, device=device)
    return _evaluate_q_policy(algorithm.q_network, config, device=device, num_episodes=num_episodes)


def _evaluate_sac(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_sac_algorithm(config, checkpoint_state, device=device)
    return _evaluate_sac_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _evaluate_td3(config: TrainConfig, checkpoint_state: CheckpointState, device: torch.device, num_episodes: int) -> MetricDict:
    algorithm = _load_td3_algorithm(config, checkpoint_state, device=device)
    return _evaluate_td3_policy(algorithm.model, config, device=device, num_episodes=num_episodes)


def _predict_a2c(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_a2c_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_ppo(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_ppo_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.policy.act(obs_tensor, deterministic=deterministic).actions
    return _format_action_output(actions, discrete=True)


def _predict_dqn(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_dqn_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    with torch.no_grad():
        actions = algorithm.q_network.act(obs_tensor, epsilon=0.0)
    return _format_action_output(actions, discrete=True)


def _predict_sac(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    algorithm = _load_sac_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.sample_actions(obs_tensor, deterministic=deterministic).actions
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


def _predict_td3(
    config: TrainConfig,
    checkpoint_state: CheckpointState,
    device: torch.device,
    obs: object,
    deterministic: bool,
) -> int | np.ndarray:
    del deterministic
    algorithm = _load_td3_algorithm(config, checkpoint_state, device=device)
    obs_tensor = _prepare_observation(obs, device=device)
    low, high = _continuous_action_bounds(config, device=device)
    with torch.no_grad():
        normalized_actions = algorithm.model.actor(obs_tensor)
        actions = _scale_continuous_actions(normalized_actions, low=low, high=high)
    return _format_action_output(actions, discrete=False)


_ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "a2c": AlgorithmSpec(
        name="a2c",
        train_fn=train_a2c,
        evaluate_fn=_evaluate_a2c,
        predict_fn=_predict_a2c,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        train_fn=train_ppo,
        evaluate_fn=_evaluate_ppo,
        predict_fn=_predict_ppo,
    ),
    "dqn": AlgorithmSpec(
        name="dqn",
        train_fn=train_dqn,
        evaluate_fn=_evaluate_dqn,
        predict_fn=_predict_dqn,
    ),
    "sac": AlgorithmSpec(
        name="sac",
        train_fn=train_sac,
        evaluate_fn=_evaluate_sac,
        predict_fn=_predict_sac,
    ),
    "td3": AlgorithmSpec(
        name="td3",
        train_fn=train_td3,
        evaluate_fn=_evaluate_td3,
        predict_fn=_predict_td3,
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm: {name!r}") from exc


def list_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(_ALGORITHM_REGISTRY.values())
