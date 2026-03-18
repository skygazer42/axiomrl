from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class MuZeroReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_shape: Sequence[int],
        action_dim: int,
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.uint8,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype

        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")
        if len(self.obs_shape) != 3:
            raise ValueError(f"expected obs_shape to be 3D (C,H,W), got {self.obs_shape!r}")
        if self.action_dim < 2:
            raise ValueError(f"action_dim must be >= 2, got {self.action_dim}")

        self.position = 0
        self.size = 0

        self.obs = torch.zeros((self.capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.next_obs = torch.zeros((self.capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.actions = torch.zeros((self.capacity,), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.policies = torch.zeros((self.capacity, self.action_dim), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros((self.capacity,), dtype=torch.int64, device=self.device)

    def reset(self) -> None:
        self.position = 0
        self.size = 0
        self.obs.zero_()
        self.next_obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.policies.zero_()
        self.steps.zero_()

    def add(
        self,
        *,
        obs: object,
        action: int,
        reward: float,
        done: bool,
        policy: object,
        next_obs: object,
        step: int,
    ) -> None:
        self.obs[self.position].copy_(torch.as_tensor(obs, device=self.device, dtype=self.obs_dtype))
        self.next_obs[self.position].copy_(torch.as_tensor(next_obs, device=self.device, dtype=self.obs_dtype))
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.dones[self.position] = 1.0 if bool(done) else 0.0
        policy_tensor = torch.as_tensor(policy, device=self.device, dtype=torch.float32).reshape(-1)
        if int(policy_tensor.shape[0]) != self.action_dim:
            raise ValueError(f"expected policy shape ({self.action_dim},), got {tuple(policy_tensor.shape)!r}")
        self.policies[self.position].copy_(policy_tensor)
        self.steps[self.position] = int(step)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, *, unroll_steps: int) -> dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if unroll_steps < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")
        if self.size < unroll_steps + 1:
            raise ValueError("not enough data to sample MuZero sequences")

        sequences: list[list[int]] = []
        max_attempts = max(100, int(batch_size) * 50)
        attempts = 0

        while len(sequences) < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("failed to sample a contiguous MuZero sequence from the replay buffer")

            if self.size < self.capacity:
                max_start = self.size - (unroll_steps + 1)
                start = int(np.random.randint(0, max_start + 1))
                indices = [start + offset for offset in range(unroll_steps + 1)]
            else:
                start = int(np.random.randint(0, self.capacity))
                indices = [(start + offset) % self.capacity for offset in range(unroll_steps + 1)]

            base_step = int(self.steps[indices[0]].item())
            if all(int(self.steps[idx].item()) == base_step + offset for offset, idx in enumerate(indices)):
                sequences.append(indices)

        index_tensor = torch.tensor(sequences, dtype=torch.int64, device=self.device)  # (B, T+1)
        transition_indices = index_tensor[:, :-1]  # (B, T)

        obs = self.obs[index_tensor[:, 0]]
        target_obs = self.obs[index_tensor[:, 1:]]
        bootstrap_obs = self.next_obs[transition_indices[:, -1]]
        actions = self.actions[transition_indices]
        rewards = self.rewards[transition_indices]
        dones = self.dones[transition_indices]
        target_policies = self.policies[index_tensor]

        return {
            "obs": obs,
            "target_obs": target_obs,
            "bootstrap_obs": bootstrap_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "target_policies": target_policies,
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "obs": self.obs.clone(),
            "next_obs": self.next_obs.clone(),
            "actions": self.actions.clone(),
            "rewards": self.rewards.clone(),
            "dones": self.dones.clone(),
            "policies": self.policies.clone(),
            "steps": self.steps.clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.obs.copy_(state_dict["obs"].to(device=self.device))
        self.next_obs.copy_(state_dict["next_obs"].to(device=self.device))
        self.actions.copy_(state_dict["actions"].to(device=self.device))
        self.rewards.copy_(state_dict["rewards"].to(device=self.device))
        self.dones.copy_(state_dict["dones"].to(device=self.device))
        self.policies.copy_(state_dict["policies"].to(device=self.device))
        self.steps.copy_(state_dict["steps"].to(device=self.device))
