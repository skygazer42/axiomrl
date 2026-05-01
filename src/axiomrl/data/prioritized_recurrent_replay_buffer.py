import math
from typing import Any

import torch

from axiomrl.data.recurrent_replay_buffer import RecurrentReplayBuffer


class PrioritizedRecurrentReplayBuffer(RecurrentReplayBuffer):
    def __init__(
        self,
        *,
        alpha: float,
        priority_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if priority_eps <= 0:
            raise ValueError(f"priority_eps must be > 0, got {priority_eps}")
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)
        self.max_priority = 1.0
        self.priorities = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        super().reset()
        self.max_priority = 1.0
        self.priorities.zero_()

    def _store_chunk(self, chunk: dict[str, Any]) -> None:
        position = self.position
        super()._store_chunk(chunk)
        self.priorities[position] = float(self.max_priority)

    def sample(self, batch_size: int, *, beta: float = 0.0) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty recurrent replay buffer")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")

        priorities = self.priorities[: self.size].clamp(min=self.priority_eps)
        if math.isclose(self.alpha, 0.0, abs_tol=1e-12):
            probs = torch.full_like(priorities, 1.0 / float(self.size))
        else:
            scaled = priorities.pow(self.alpha)
            probs = scaled / scaled.sum()

        indices = torch.multinomial(probs, int(batch_size), replacement=True)
        weights = (float(self.size) * probs[indices]).pow(-float(beta))
        weights = weights / (weights.max() + 1e-12)

        batch = {
            "obs": self.obs[indices].permute(1, 0, *range(2, self.obs.ndim)),
            "actions": self.actions[indices].permute(1, 0),
            "rewards": self.rewards[indices].permute(1, 0),
            "next_obs": self.next_obs[indices].permute(1, 0, *range(2, self.next_obs.ndim)),
            "dones": self.dones[indices].permute(1, 0),
            "episode_starts": self.episode_starts[indices].permute(1, 0),
            "mask": self.mask[indices].permute(1, 0),
            "initial_h": self.initial_h[indices].permute(1, 0, 2),
            "initial_c": self.initial_c[indices].permute(1, 0, 2),
            "indices": indices,
            "weights": weights,
        }
        return batch

    def update_priorities(self, indices: torch.Tensor, priorities: object) -> None:
        if self.size == 0:
            return
        index_tensor = torch.as_tensor(indices, device=self.device, dtype=torch.int64)
        priority_tensor = torch.as_tensor(priorities, device=self.device, dtype=torch.float32).abs() + self.priority_eps
        self.priorities[index_tensor] = priority_tensor
        self.max_priority = max(self.max_priority, float(priority_tensor.max().detach().cpu().item()))

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "alpha": self.alpha,
                "priority_eps": self.priority_eps,
                "max_priority": self.max_priority,
                "priorities": self.priorities.clone(),
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.priority_eps = float(state_dict.get("priority_eps", self.priority_eps))
        self.max_priority = float(state_dict.get("max_priority", 1.0))
        self.priorities.copy_(state_dict.get("priorities", torch.zeros_like(self.priorities)).to(device=self.device))
