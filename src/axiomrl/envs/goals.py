from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


GOAL_OBSERVATION_KEY = "observation"
ACHIEVED_GOAL_KEY = "achieved_goal"
DESIRED_GOAL_KEY = "desired_goal"
POINT_GOAL_ENV_ID = "RL-PointGoal1D-v0"


@dataclass(frozen=True, slots=True)
class GoalSpaceSpec:
    observation_dim: int
    goal_dim: int

    @property
    def flat_observation_dim(self) -> int:
        return self.observation_dim + self.goal_dim


class PointGoal1DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_episode_steps: int = 50,
        goal_threshold: float = 0.05,
        action_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self.goal_threshold = float(goal_threshold)
        self.action_scale = float(action_scale)

        self.observation_space = gym.spaces.Dict(
            {
                GOAL_OBSERVATION_KEY: gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
                ACHIEVED_GOAL_KEY: gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
                DESIRED_GOAL_KEY: gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._state = np.zeros((1,), dtype=np.float32)
        self._goal = np.zeros((1,), dtype=np.float32)
        self._step_count = 0

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = self._state.astype(np.float32, copy=True)
        goal = self._goal.astype(np.float32, copy=True)
        return {
            GOAL_OBSERVATION_KEY: state,
            ACHIEVED_GOAL_KEY: state.copy(),
            DESIRED_GOAL_KEY: goal,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        del options
        self._step_count = 0
        self._state = self.np_random.uniform(low=-1.0, high=1.0, size=(1,)).astype(np.float32)
        self._goal = self.np_random.uniform(low=-1.0, high=1.0, size=(1,)).astype(np.float32)
        if np.allclose(self._state, self._goal, atol=self.goal_threshold):
            self._goal = np.clip(self._goal + 0.5, -1.0, 1.0).astype(np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        self._state = np.clip(self._state + clipped_action * self.action_scale, -2.0, 2.0).astype(np.float32)
        self._step_count += 1

        obs = self._get_obs()
        reward = float(self.compute_reward(obs[ACHIEVED_GOAL_KEY], obs[DESIRED_GOAL_KEY], {}))
        terminated = bool(self.compute_terminated(obs[ACHIEVED_GOAL_KEY], obs[DESIRED_GOAL_KEY], {}))
        truncated = self._step_count >= self.max_episode_steps and not terminated
        return obs, reward, terminated, truncated, {}

    def compute_reward(self, achieved_goal: object, desired_goal: object, info: object) -> float | np.ndarray:
        del info
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        reward = np.where(distance <= self.goal_threshold, 0.0, -1.0).astype(np.float32)
        if reward.ndim == 0:
            return float(reward.item())
        return reward

    def compute_terminated(self, achieved_goal: object, desired_goal: object, info: object) -> bool | np.ndarray:
        del info
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        terminated = distance <= self.goal_threshold
        if np.asarray(terminated).ndim == 0:
            return bool(np.asarray(terminated).item())
        return terminated.astype(bool)

    def compute_truncated(self, achieved_goal: object, desired_goal: object, info: object) -> bool | np.ndarray:
        del achieved_goal, desired_goal, info
        return False


def register_builtin_goal_envs() -> None:
    try:
        gym.spec(POINT_GOAL_ENV_ID)
    except gym.error.Error:
        gym.register(
            id=POINT_GOAL_ENV_ID,
            entry_point=PointGoal1DEnv,
        )


def is_goal_observation_space(space: gym.Space[Any]) -> bool:
    if not isinstance(space, gym.spaces.Dict):
        return False
    required = {GOAL_OBSERVATION_KEY, ACHIEVED_GOAL_KEY, DESIRED_GOAL_KEY}
    if not required.issubset(space.spaces):
        return False
    return all(isinstance(space.spaces[key], gym.spaces.Box) for key in required)


def infer_goal_space_spec(space: gym.Space[Any]) -> GoalSpaceSpec:
    if not is_goal_observation_space(space):
        raise TypeError(f"expected a goal-conditioned Dict space, got {type(space)!r}")

    assert isinstance(space, gym.spaces.Dict)
    observation_space = space.spaces[GOAL_OBSERVATION_KEY]
    desired_goal_space = space.spaces[DESIRED_GOAL_KEY]
    if observation_space.shape is None or len(observation_space.shape) != 1:
        raise ValueError(f"expected flat 1D observation goal component, got shape={observation_space.shape!r}")
    if desired_goal_space.shape is None or len(desired_goal_space.shape) != 1:
        raise ValueError(f"expected flat 1D desired_goal component, got shape={desired_goal_space.shape!r}")

    return GoalSpaceSpec(
        observation_dim=int(observation_space.shape[0]),
        goal_dim=int(desired_goal_space.shape[0]),
    )


def split_goal_observation(observation: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        obs = np.asarray(observation[GOAL_OBSERVATION_KEY], dtype=np.float32)
        achieved_goal = np.asarray(observation[ACHIEVED_GOAL_KEY], dtype=np.float32)
        desired_goal = np.asarray(observation[DESIRED_GOAL_KEY], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"missing goal observation key: {exc.args[0]!r}") from exc
    return obs, achieved_goal, desired_goal


def flatten_goal_observation(observation: Mapping[str, object], *, desired_goal: object | None = None) -> np.ndarray:
    obs, _, original_desired_goal = split_goal_observation(observation)
    desired_goal_array = original_desired_goal if desired_goal is None else np.asarray(desired_goal, dtype=np.float32)
    return np.concatenate([obs, desired_goal_array], axis=-1).astype(np.float32)


def goal_env_compute_reward(env: gym.Env, *, achieved_goal: object, desired_goal: object, info: object | None = None) -> float:
    compute_reward = getattr(env.unwrapped, "compute_reward", None)
    if not callable(compute_reward):
        raise AttributeError(f"env {type(env.unwrapped)!r} does not expose compute_reward(...)")
    reward = compute_reward(achieved_goal, desired_goal, {} if info is None else info)
    reward_array = np.asarray(reward, dtype=np.float32)
    return float(reward_array.reshape(-1)[0].item())


def goal_env_compute_done(
    env: gym.Env,
    *,
    achieved_goal: object,
    desired_goal: object,
    info: object | None = None,
    fallback_done: bool | None = None,
    fallback_truncated: bool = False,
) -> float:
    compute_terminated = getattr(env.unwrapped, "compute_terminated", None)
    compute_truncated = getattr(env.unwrapped, "compute_truncated", None)
    if not callable(compute_terminated) and not callable(compute_truncated):
        if fallback_done is not None:
            return float(fallback_done)
        return float(fallback_truncated)

    info_payload = {} if info is None else info
    terminated = False
    truncated = False
    if callable(compute_terminated):
        terminated = bool(np.asarray(compute_terminated(achieved_goal, desired_goal, info_payload)).reshape(-1)[0].item())
    if callable(compute_truncated):
        truncated = bool(np.asarray(compute_truncated(achieved_goal, desired_goal, info_payload)).reshape(-1)[0].item())
    return float(terminated or truncated or fallback_truncated)
