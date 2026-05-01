import math
from collections.abc import Sequence

import torch


class PrioritizedReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        alpha: float,
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype | None = None,
        priority_eps: float = 1e-6,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if priority_eps <= 0:
            raise ValueError(f"priority_eps must be > 0, got {priority_eps}")

        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype or (torch.int64 if not self.action_shape else torch.float32)

        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)

        self.position = 0
        self.size = 0
        self.max_priority = 1.0

        self.obs = torch.zeros((capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.actions = torch.zeros((capacity, *self.action_shape), dtype=self.action_dtype, device=self.device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((capacity, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.priorities = torch.zeros((capacity,), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_obs.zero_()
        self.dones.zero_()
        self.priorities.zero_()

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

        self.priorities[self.position] = float(self.max_priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, beta: float) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")

        priorities = self.priorities[: self.size].clamp(min=self.priority_eps)
        if math.isclose(self.alpha, 0.0):
            probs = torch.full_like(priorities, 1.0 / float(self.size))
        else:
            scaled = priorities.pow(self.alpha)
            probs = scaled / scaled.sum()

        indices = torch.multinomial(probs, int(batch_size), replacement=True)
        sampled_actions = self.actions[indices]
        if not self.action_shape:
            sampled_actions = sampled_actions.reshape(batch_size)

        weights = (float(self.size) * probs[indices]).pow(-float(beta))
        weights = weights / (weights.max() + 1e-12)

        return {
            "obs": self.obs[indices],
            "actions": sampled_actions,
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(self, indices: torch.Tensor, priorities: object) -> None:
        if self.size == 0:
            return
        index_tensor = torch.as_tensor(indices, device=self.device, dtype=torch.int64)
        priority_tensor = torch.as_tensor(priorities, device=self.device, dtype=torch.float32).abs() + self.priority_eps
        self.priorities[index_tensor] = priority_tensor
        self.max_priority = max(self.max_priority, float(priority_tensor.max().detach().cpu().item()))

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "alpha": self.alpha,
            "priority_eps": self.priority_eps,
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
            # Checkpoints should live on CPU so saving does not duplicate the full buffer on GPU.
            "obs": self.obs.detach().cpu().clone(),
            "actions": self.actions.detach().cpu().clone(),
            "rewards": self.rewards.detach().cpu().clone(),
            "next_obs": self.next_obs.detach().cpu().clone(),
            "dones": self.dones.detach().cpu().clone(),
            "priorities": self.priorities.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.max_priority = float(state_dict.get("max_priority", 1.0))
        self.obs.copy_(state_dict["obs"].to(device=self.device))
        self.actions.copy_(state_dict["actions"].to(device=self.device))
        self.rewards.copy_(state_dict["rewards"].to(device=self.device))
        self.next_obs.copy_(state_dict["next_obs"].to(device=self.device))
        self.dones.copy_(state_dict["dones"].to(device=self.device))
        self.priorities.copy_(state_dict.get("priorities", torch.zeros_like(self.priorities)).to(device=self.device))
