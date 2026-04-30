from __future__ import annotations

from collections.abc import Sequence

import torch


class ReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype | None = None,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype or (torch.int64 if not self.action_shape else torch.float32)
        self.position = 0
        self.size = 0

        self.obs = torch.zeros((capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.actions = torch.zeros((capacity, *self.action_shape), dtype=self.action_dtype, device=self.device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.position = 0
        self.size = 0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_obs.zero_()
        self.dones.zero_()

    def add(
        self,
        *,
        obs: object,
        actions: object,
        rewards: object,
        next_obs: object,
        dones: object,
    ) -> None:
        self.obs[self.position].copy_(torch.as_tensor(obs, device=self.device, dtype=self.obs_dtype))
        self.actions[self.position].copy_(torch.as_tensor(actions, device=self.device, dtype=self.action_dtype))
        self.rewards[self.position].copy_(torch.as_tensor(rewards, device=self.device, dtype=torch.float32))
        self.next_obs[self.position].copy_(torch.as_tensor(next_obs, device=self.device, dtype=self.obs_dtype))
        self.dones[self.position].copy_(torch.as_tensor(dones, device=self.device, dtype=torch.float32))

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        sampled_actions = self.actions[indices]
        if not self.action_shape:
            sampled_actions = sampled_actions.reshape(batch_size)

        return {
            "obs": self.obs[indices],
            "actions": sampled_actions,
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "position": self.position,
            "size": self.size,
            # Checkpoints should live on CPU so saving does not duplicate the full buffer on GPU.
            "obs": self.obs.detach().cpu().clone(),
            "actions": self.actions.detach().cpu().clone(),
            "rewards": self.rewards.detach().cpu().clone(),
            "next_obs": self.next_obs.detach().cpu().clone(),
            "dones": self.dones.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.obs.copy_(state_dict["obs"].to(device=self.device))
        self.actions.copy_(state_dict["actions"].to(device=self.device))
        self.rewards.copy_(state_dict["rewards"].to(device=self.device))
        self.next_obs.copy_(state_dict["next_obs"].to(device=self.device))
        self.dones.copy_(state_dict["dones"].to(device=self.device))
