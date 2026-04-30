from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from rl_training.experiment.config import TrainConfig
from rl_training.envs.factory import build_env


def _infer_discrete_env_spaces(config: TrainConfig) -> tuple[tuple[int, ...], int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) not in (1, 3):
            raise ValueError(f"expected flat 1D or channel-first image observations, got shape={obs_space.shape!r}")
        return tuple(int(dim) for dim in obs_space.shape), int(action_space.n)
    finally:
        env.close()


def _infer_continuous_env_spaces(config: TrainConfig) -> tuple[int, int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
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
        env.close()


def _infer_image_continuous_env_spaces(config: TrainConfig) -> tuple[tuple[int, ...], int]:
    env = build_env(config, 0)
    try:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space: {type(obs_space)!r}")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"unsupported action space: {type(action_space)!r}")
        if obs_space.shape is None or len(obs_space.shape) != 3:
            raise ValueError(f"expected channel-first image observations, got shape={obs_space.shape!r}")
        if action_space.shape is None or len(action_space.shape) != 1:
            raise ValueError(f"expected flat 1D actions, got shape={action_space.shape!r}")
        return tuple(int(dim) for dim in obs_space.shape), int(action_space.shape[0])
    finally:
        env.close()


def _continuous_action_bounds(config: TrainConfig, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    env = build_env(config, 0, evaluation=True)
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
    if obs_tensor.ndim in (1, 3):
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
