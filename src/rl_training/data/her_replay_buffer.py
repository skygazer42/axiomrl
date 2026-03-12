from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_training.envs.goals import (
    flatten_goal_observation,
    goal_env_compute_done,
    goal_env_compute_reward,
    split_goal_observation,
)


class HERReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        obs_shape: Sequence[int],
        goal_shape: Sequence[int],
        action_shape: Sequence[int],
        her_ratio: float = 0.8,
        goal_selection_strategy: str = "future",
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
    ) -> None:
        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if int(num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not 0.0 <= float(her_ratio) <= 1.0:
            raise ValueError(f"her_ratio must be in [0, 1], got {her_ratio}")
        if goal_selection_strategy != "future":
            raise ValueError(f"unsupported goal_selection_strategy: {goal_selection_strategy!r}")

        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.goal_shape = tuple(int(dim) for dim in goal_shape)
        self.action_shape = tuple(int(dim) for dim in action_shape)
        self.her_ratio = float(her_ratio)
        self.goal_selection_strategy = goal_selection_strategy
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype

        self._episodes: list[dict[str, Any]] = []
        self._current_episodes: list[dict[str, list[Any]]] = [self._empty_episode() for _ in range(self.num_envs)]
        self._num_transitions = 0

    def _empty_episode(self) -> dict[str, list[Any]]:
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "achieved_goals": [],
            "desired_goals": [],
            "next_achieved_goals": [],
            "terminateds": [],
            "truncateds": [],
        }

    def reset(self) -> None:
        self._episodes.clear()
        self._current_episodes = [self._empty_episode() for _ in range(self.num_envs)]
        self._num_transitions = 0

    def add(
        self,
        *,
        env_index: int,
        obs: dict[str, object],
        actions: object,
        rewards: float,
        next_obs: dict[str, object],
        terminated: bool,
        truncated: bool,
    ) -> None:
        episode = self._current_episodes[env_index]
        observation, achieved_goal, desired_goal = split_goal_observation(obs)
        next_observation, next_achieved_goal, _ = split_goal_observation(next_obs)

        episode["observations"].append(np.asarray(observation, dtype=np.float32))
        episode["actions"].append(np.asarray(actions, dtype=np.float32))
        episode["rewards"].append(float(rewards))
        episode["next_observations"].append(np.asarray(next_observation, dtype=np.float32))
        episode["achieved_goals"].append(np.asarray(achieved_goal, dtype=np.float32))
        episode["desired_goals"].append(np.asarray(desired_goal, dtype=np.float32))
        episode["next_achieved_goals"].append(np.asarray(next_achieved_goal, dtype=np.float32))
        episode["terminateds"].append(bool(terminated))
        episode["truncateds"].append(bool(truncated))

        if terminated or truncated:
            self._finalize_episode(env_index)

    def _finalize_episode(self, env_index: int) -> None:
        episode_lists = self._current_episodes[env_index]
        if not episode_lists["actions"]:
            self._current_episodes[env_index] = self._empty_episode()
            return

        episode_length = len(episode_lists["actions"])
        frozen_episode = {
            "observations": torch.stack(
                [torch.as_tensor(item, dtype=self.obs_dtype, device=self.device) for item in episode_lists["observations"]]
            ),
            "actions": torch.stack(
                [torch.as_tensor(item, dtype=self.action_dtype, device=self.device) for item in episode_lists["actions"]]
            ),
            "rewards": torch.as_tensor(episode_lists["rewards"], dtype=torch.float32, device=self.device),
            "next_observations": torch.stack(
                [torch.as_tensor(item, dtype=self.obs_dtype, device=self.device) for item in episode_lists["next_observations"]]
            ),
            "achieved_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode_lists["achieved_goals"]]
            ),
            "desired_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode_lists["desired_goals"]]
            ),
            "next_achieved_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode_lists["next_achieved_goals"]]
            ),
            "terminateds": torch.as_tensor(episode_lists["terminateds"], dtype=torch.float32, device=self.device),
            "truncateds": torch.as_tensor(episode_lists["truncateds"], dtype=torch.float32, device=self.device),
            "length": episode_length,
        }
        self._episodes.append(frozen_episode)
        self._num_transitions += episode_length
        self._current_episodes[env_index] = self._empty_episode()

        while self._episodes and self._num_transitions > self.capacity:
            removed = self._episodes.pop(0)
            self._num_transitions -= int(removed["length"])

    def sample(self, batch_size: int, *, env: gym.Env) -> dict[str, torch.Tensor]:
        if not self._episodes:
            raise ValueError("cannot sample from an empty HER replay buffer")

        episode_lengths = np.asarray([int(episode["length"]) for episode in self._episodes], dtype=np.float64)
        episode_probabilities = episode_lengths / episode_lengths.sum()
        episode_indices = np.random.choice(len(self._episodes), size=batch_size, p=episode_probabilities)

        obs_batch: list[torch.Tensor] = []
        action_batch: list[torch.Tensor] = []
        reward_batch: list[float] = []
        next_obs_batch: list[torch.Tensor] = []
        done_batch: list[float] = []

        for episode_index in episode_indices:
            episode = self._episodes[int(episode_index)]
            episode_length = int(episode["length"])
            transition_index = int(np.random.randint(0, episode_length))
            relabel = self.her_ratio > 0.0 and np.random.random() < self.her_ratio

            observation = episode["observations"][transition_index]
            next_observation = episode["next_observations"][transition_index]
            desired_goal = episode["desired_goals"][transition_index]
            reward = float(episode["rewards"][transition_index].item())
            done = float((episode["terminateds"][transition_index] + episode["truncateds"][transition_index]).clamp(max=1.0).item())

            if relabel:
                future_index = int(np.random.randint(transition_index, episode_length))
                desired_goal = episode["next_achieved_goals"][future_index]
                reward = goal_env_compute_reward(
                    env,
                    achieved_goal=episode["next_achieved_goals"][transition_index].detach().cpu().numpy(),
                    desired_goal=desired_goal.detach().cpu().numpy(),
                    info={},
                )
                done = goal_env_compute_done(
                    env,
                    achieved_goal=episode["next_achieved_goals"][transition_index].detach().cpu().numpy(),
                    desired_goal=desired_goal.detach().cpu().numpy(),
                    info={},
                    fallback_truncated=bool(episode["truncateds"][transition_index].item()),
                )

            obs_batch.append(
                torch.as_tensor(
                    flatten_goal_observation(
                        {
                            "observation": observation.detach().cpu().numpy(),
                            "achieved_goal": episode["achieved_goals"][transition_index].detach().cpu().numpy(),
                            "desired_goal": desired_goal.detach().cpu().numpy(),
                        }
                    ),
                    dtype=self.obs_dtype,
                    device=self.device,
                )
            )
            next_obs_batch.append(
                torch.as_tensor(
                    flatten_goal_observation(
                        {
                            "observation": next_observation.detach().cpu().numpy(),
                            "achieved_goal": episode["next_achieved_goals"][transition_index].detach().cpu().numpy(),
                            "desired_goal": desired_goal.detach().cpu().numpy(),
                        }
                    ),
                    dtype=self.obs_dtype,
                    device=self.device,
                )
            )
            action_batch.append(episode["actions"][transition_index].to(device=self.device))
            reward_batch.append(float(reward))
            done_batch.append(float(done))

        return {
            "obs": torch.stack(obs_batch),
            "actions": torch.stack(action_batch),
            "rewards": torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device),
            "next_obs": torch.stack(next_obs_batch),
            "dones": torch.as_tensor(done_batch, dtype=torch.float32, device=self.device),
        }

    def __len__(self) -> int:
        return self._num_transitions

    def _freeze_partial_episode(self, episode: dict[str, list[Any]]) -> dict[str, Any]:
        if not episode["actions"]:
            return {"length": 0}
        return {
            "observations": torch.stack(
                [torch.as_tensor(item, dtype=self.obs_dtype, device=self.device) for item in episode["observations"]]
            ),
            "actions": torch.stack(
                [torch.as_tensor(item, dtype=self.action_dtype, device=self.device) for item in episode["actions"]]
            ),
            "rewards": torch.as_tensor(episode["rewards"], dtype=torch.float32, device=self.device),
            "next_observations": torch.stack(
                [torch.as_tensor(item, dtype=self.obs_dtype, device=self.device) for item in episode["next_observations"]]
            ),
            "achieved_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode["achieved_goals"]]
            ),
            "desired_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode["desired_goals"]]
            ),
            "next_achieved_goals": torch.stack(
                [torch.as_tensor(item, dtype=torch.float32, device=self.device) for item in episode["next_achieved_goals"]]
            ),
            "terminateds": torch.as_tensor(episode["terminateds"], dtype=torch.float32, device=self.device),
            "truncateds": torch.as_tensor(episode["truncateds"], dtype=torch.float32, device=self.device),
            "length": len(episode["actions"]),
        }

    def _thaw_partial_episode(self, frozen_episode: dict[str, Any]) -> dict[str, list[Any]]:
        if int(frozen_episode.get("length", 0)) == 0:
            return self._empty_episode()

        def _to_numpy_list(tensor_key: str) -> list[np.ndarray]:
            tensor = frozen_episode[tensor_key]
            return [tensor[index].detach().cpu().numpy() for index in range(int(frozen_episode["length"]))]

        rewards_tensor = frozen_episode["rewards"]
        terminateds_tensor = frozen_episode["terminateds"]
        truncateds_tensor = frozen_episode["truncateds"]
        return {
            "observations": _to_numpy_list("observations"),
            "actions": _to_numpy_list("actions"),
            "rewards": [float(rewards_tensor[index].item()) for index in range(int(frozen_episode["length"]))],
            "next_observations": _to_numpy_list("next_observations"),
            "achieved_goals": _to_numpy_list("achieved_goals"),
            "desired_goals": _to_numpy_list("desired_goals"),
            "next_achieved_goals": _to_numpy_list("next_achieved_goals"),
            "terminateds": [bool(terminateds_tensor[index].item()) for index in range(int(frozen_episode["length"]))],
            "truncateds": [bool(truncateds_tensor[index].item()) for index in range(int(frozen_episode["length"]))],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "obs_shape": self.obs_shape,
            "goal_shape": self.goal_shape,
            "action_shape": self.action_shape,
            "her_ratio": self.her_ratio,
            "goal_selection_strategy": self.goal_selection_strategy,
            "num_transitions": self._num_transitions,
            "episodes": [
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in episode.items()
                }
                for episode in self._episodes
            ],
            "current_episodes": [self._freeze_partial_episode(episode) for episode in self._current_episodes],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._episodes = []
        for episode in state_dict.get("episodes", ()):
            self._episodes.append(
                {
                    key: value.to(device=self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in episode.items()
                }
            )
        self._current_episodes = [self._thaw_partial_episode(episode) for episode in state_dict.get("current_episodes", ())]
        if len(self._current_episodes) != self.num_envs:
            self._current_episodes = [self._empty_episode() for _ in range(self.num_envs)]
        self._num_transitions = int(state_dict.get("num_transitions", sum(int(episode["length"]) for episode in self._episodes)))
