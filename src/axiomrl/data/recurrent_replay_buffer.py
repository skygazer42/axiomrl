from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class RecurrentReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        obs_shape: Sequence[int],
        sequence_length: int,
        hidden_size: int,
        num_layers: int = 1,
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.position = 0
        self.size = 0
        self.num_transitions = 0

        self.obs = torch.zeros(
            (self.capacity, self.sequence_length, *self.obs_shape),
            dtype=self.obs_dtype,
            device=self.device,
        )
        self.actions = torch.zeros((self.capacity, self.sequence_length), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((self.capacity, self.sequence_length), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros(
            (self.capacity, self.sequence_length, *self.obs_shape),
            dtype=self.obs_dtype,
            device=self.device,
        )
        self.dones = torch.zeros((self.capacity, self.sequence_length), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.zeros((self.capacity, self.sequence_length), dtype=torch.float32, device=self.device)
        self.mask = torch.zeros((self.capacity, self.sequence_length), dtype=torch.float32, device=self.device)
        self.initial_h = torch.zeros(
            (self.capacity, self.num_layers, self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.initial_c = torch.zeros(
            (self.capacity, self.num_layers, self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._active_chunks: list[dict[str, Any]] = [self._empty_active_chunk() for _ in range(self.num_envs)]

    def _empty_active_chunk(self) -> dict[str, Any]:
        return {
            "length": 0,
            "obs": torch.zeros((self.sequence_length, *self.obs_shape), dtype=self.obs_dtype, device=self.device),
            "actions": torch.zeros((self.sequence_length,), dtype=torch.int64, device=self.device),
            "rewards": torch.zeros((self.sequence_length,), dtype=torch.float32, device=self.device),
            "next_obs": torch.zeros((self.sequence_length, *self.obs_shape), dtype=self.obs_dtype, device=self.device),
            "dones": torch.zeros((self.sequence_length,), dtype=torch.float32, device=self.device),
            "episode_starts": torch.zeros((self.sequence_length,), dtype=torch.float32, device=self.device),
            "mask": torch.zeros((self.sequence_length,), dtype=torch.float32, device=self.device),
            "initial_h": torch.zeros((self.num_layers, self.hidden_size), dtype=torch.float32, device=self.device),
            "initial_c": torch.zeros((self.num_layers, self.hidden_size), dtype=torch.float32, device=self.device),
        }

    def reset(self) -> None:
        self.position = 0
        self.size = 0
        self.num_transitions = 0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_obs.zero_()
        self.dones.zero_()
        self.episode_starts.zero_()
        self.mask.zero_()
        self.initial_h.zero_()
        self.initial_c.zero_()
        self._active_chunks = [self._empty_active_chunk() for _ in range(self.num_envs)]

    def clear_active_chunks(self) -> None:
        dropped_transitions = sum(int(chunk["length"]) for chunk in self._active_chunks)
        self.num_transitions = max(0, self.num_transitions - dropped_transitions)
        self._active_chunks = [self._empty_active_chunk() for _ in range(self.num_envs)]

    def _normalize_state(self, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        hidden_tensor = torch.as_tensor(hidden, dtype=torch.float32, device=self.device)
        cell_tensor = torch.as_tensor(cell, dtype=torch.float32, device=self.device)
        if hidden_tensor.ndim == 3:
            if hidden_tensor.shape[1] != 1 or cell_tensor.shape[1] != 1:
                raise ValueError("initial_state for RecurrentReplayBuffer.add must describe a single environment")
            hidden_tensor = hidden_tensor.squeeze(1)
            cell_tensor = cell_tensor.squeeze(1)
        return hidden_tensor, cell_tensor

    def _store_chunk(self, chunk: dict[str, Any]) -> None:
        self.obs[self.position].copy_(chunk["obs"])
        self.actions[self.position].copy_(chunk["actions"])
        self.rewards[self.position].copy_(chunk["rewards"])
        self.next_obs[self.position].copy_(chunk["next_obs"])
        self.dones[self.position].copy_(chunk["dones"])
        self.episode_starts[self.position].copy_(chunk["episode_starts"])
        self.mask[self.position].copy_(chunk["mask"])
        self.initial_h[self.position].copy_(chunk["initial_h"])
        self.initial_c[self.position].copy_(chunk["initial_c"])
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _finalize_active_chunk(self, env_index: int) -> None:
        chunk = self._active_chunks[env_index]
        if int(chunk["length"]) == 0:
            return
        self._store_chunk(chunk)
        self._active_chunks[env_index] = self._empty_active_chunk()

    def add(
        self,
        *,
        env_index: int,
        obs: object,
        actions: object,
        rewards: object,
        next_obs: object,
        dones: object,
        episode_start: object,
        initial_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index must be in [0, {self.num_envs}), got {env_index}")

        chunk = self._active_chunks[env_index]
        position = int(chunk["length"])
        if position == 0:
            initial_h, initial_c = self._normalize_state(initial_state)
            chunk["initial_h"].copy_(initial_h)
            chunk["initial_c"].copy_(initial_c)

        chunk["obs"][position].copy_(torch.as_tensor(obs, dtype=self.obs_dtype, device=self.device))
        chunk["actions"][position].copy_(torch.as_tensor(actions, dtype=torch.int64, device=self.device))
        chunk["rewards"][position].copy_(torch.as_tensor(rewards, dtype=torch.float32, device=self.device))
        chunk["next_obs"][position].copy_(torch.as_tensor(next_obs, dtype=self.obs_dtype, device=self.device))
        chunk["dones"][position].copy_(torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        chunk["episode_starts"][position].copy_(torch.as_tensor(episode_start, dtype=torch.float32, device=self.device))
        chunk["mask"][position] = 1.0
        chunk["length"] = position + 1
        self.num_transitions += 1

        if bool(torch.as_tensor(dones).item()) or int(chunk["length"]) >= self.sequence_length:
            self._finalize_active_chunk(env_index)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty recurrent replay buffer")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[indices].permute(1, 0, *range(2, self.obs.ndim)),
            "actions": self.actions[indices].permute(1, 0),
            "rewards": self.rewards[indices].permute(1, 0),
            "next_obs": self.next_obs[indices].permute(1, 0, *range(2, self.next_obs.ndim)),
            "dones": self.dones[indices].permute(1, 0),
            "episode_starts": self.episode_starts[indices].permute(1, 0),
            "mask": self.mask[indices].permute(1, 0),
            "initial_h": self.initial_h[indices].permute(1, 0, 2),
            "initial_c": self.initial_c[indices].permute(1, 0, 2),
        }

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, Any]:
        active_chunks: list[dict[str, Any]] = []
        for chunk in self._active_chunks:
            active_chunks.append(
                {
                    "length": int(chunk["length"]),
                    "obs": chunk["obs"].clone(),
                    "actions": chunk["actions"].clone(),
                    "rewards": chunk["rewards"].clone(),
                    "next_obs": chunk["next_obs"].clone(),
                    "dones": chunk["dones"].clone(),
                    "episode_starts": chunk["episode_starts"].clone(),
                    "mask": chunk["mask"].clone(),
                    "initial_h": chunk["initial_h"].clone(),
                    "initial_c": chunk["initial_c"].clone(),
                }
            )
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "obs_shape": self.obs_shape,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "position": self.position,
            "size": self.size,
            "num_transitions": self.num_transitions,
            "obs": self.obs.clone(),
            "actions": self.actions.clone(),
            "rewards": self.rewards.clone(),
            "next_obs": self.next_obs.clone(),
            "dones": self.dones.clone(),
            "episode_starts": self.episode_starts.clone(),
            "mask": self.mask.clone(),
            "initial_h": self.initial_h.clone(),
            "initial_c": self.initial_c.clone(),
            "active_chunks": active_chunks,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.num_transitions = int(state_dict.get("num_transitions", 0))
        self.obs.copy_(state_dict["obs"].to(device=self.device))
        self.actions.copy_(state_dict["actions"].to(device=self.device))
        self.rewards.copy_(state_dict["rewards"].to(device=self.device))
        self.next_obs.copy_(state_dict["next_obs"].to(device=self.device))
        self.dones.copy_(state_dict["dones"].to(device=self.device))
        self.episode_starts.copy_(state_dict["episode_starts"].to(device=self.device))
        self.mask.copy_(state_dict["mask"].to(device=self.device))
        self.initial_h.copy_(state_dict["initial_h"].to(device=self.device))
        self.initial_c.copy_(state_dict["initial_c"].to(device=self.device))

        loaded_active_chunks = state_dict.get("active_chunks", ())
        self._active_chunks = []
        for env_index in range(self.num_envs):
            if env_index < len(loaded_active_chunks):
                chunk_state = loaded_active_chunks[env_index]
                chunk = self._empty_active_chunk()
                chunk["length"] = int(chunk_state["length"])
                chunk["obs"].copy_(chunk_state["obs"].to(device=self.device))
                chunk["actions"].copy_(chunk_state["actions"].to(device=self.device))
                chunk["rewards"].copy_(chunk_state["rewards"].to(device=self.device))
                chunk["next_obs"].copy_(chunk_state["next_obs"].to(device=self.device))
                chunk["dones"].copy_(chunk_state["dones"].to(device=self.device))
                chunk["episode_starts"].copy_(chunk_state["episode_starts"].to(device=self.device))
                chunk["mask"].copy_(chunk_state["mask"].to(device=self.device))
                chunk["initial_h"].copy_(chunk_state["initial_h"].to(device=self.device))
                chunk["initial_c"].copy_(chunk_state["initial_c"].to(device=self.device))
                self._active_chunks.append(chunk)
            else:
                self._active_chunks.append(self._empty_active_chunk())
