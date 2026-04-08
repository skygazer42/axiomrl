from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rl_training.envs.atari import ensure_atari_env_registered


def _stack_samples(samples: list[np.ndarray]) -> np.ndarray:
    first = samples[0]
    if first.ndim == 0:
        return np.asarray(samples, dtype=first.dtype)
    return np.stack(samples, axis=0).astype(first.dtype, copy=False)


def _sample_action(space: gym.Space[Any]) -> np.ndarray:
    sampled = np.asarray(space.sample())
    if sampled.ndim == 0:
        return sampled.astype(np.int64, copy=False)
    if np.issubdtype(sampled.dtype, np.floating):
        return sampled.astype(np.float32, copy=False)
    return sampled.astype(np.int64, copy=False)


def _zero_action(space: gym.Space[Any]) -> np.ndarray:
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=np.float32)
    if isinstance(space, gym.spaces.Discrete):
        return np.asarray(0, dtype=np.int64)

    sampled = np.asarray(space.sample())
    if sampled.ndim == 0:
        return np.asarray(0, dtype=np.int64)
    if np.issubdtype(sampled.dtype, np.floating):
        return np.zeros_like(sampled, dtype=np.float32)
    return np.zeros_like(sampled, dtype=np.int64)


def collect_random_transition_dataset(
    env: gym.Env[Any, Any],
    *,
    num_steps: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    total_steps = int(num_steps)
    if total_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    if seed is not None:
        env.action_space.seed(int(seed))
        obs, _ = env.reset(seed=int(seed))
    else:
        obs, _ = env.reset()

    obs_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    reward_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    next_action_parts: list[np.ndarray] = []

    action = _sample_action(env.action_space)

    for _ in range(total_steps):
        next_obs, reward, terminated, truncated, _ = env.step(action.item() if action.ndim == 0 else action)
        done = bool(terminated or truncated)
        next_action = _zero_action(env.action_space) if done else _sample_action(env.action_space)

        obs_parts.append(np.asarray(obs, dtype=np.float32))
        action_parts.append(action.copy())
        reward_parts.append(np.asarray(float(reward), dtype=np.float32))
        next_obs_parts.append(np.asarray(next_obs, dtype=np.float32))
        done_parts.append(np.asarray(float(done), dtype=np.float32))
        next_action_parts.append(next_action.copy())

        if done:
            obs, _ = env.reset()
            action = _sample_action(env.action_space)
        else:
            obs = next_obs
            action = next_action

    return {
        "obs": _stack_samples(obs_parts).astype(np.float32, copy=False),
        "actions": _stack_samples(action_parts),
        "rewards": _stack_samples(reward_parts).astype(np.float32, copy=False),
        "next_obs": _stack_samples(next_obs_parts).astype(np.float32, copy=False),
        "dones": _stack_samples(done_parts).astype(np.float32, copy=False),
        "next_actions": _stack_samples(next_action_parts),
    }


def save_transition_dataset_npz(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **payload)
    return destination


def export_random_transition_dataset(
    env_id: str,
    path: str | Path,
    *,
    num_steps: int,
    seed: int = 0,
    env_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    ensure_atari_env_registered(env_id=env_id)
    env = gym.make(env_id, **dict(env_kwargs or {}))
    try:
        payload = collect_random_transition_dataset(env, num_steps=num_steps, seed=seed)
    finally:
        env.close()
    return save_transition_dataset_npz(path, payload)
