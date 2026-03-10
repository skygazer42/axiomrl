from __future__ import annotations

from typing import Any

import torch


def _to_1d_float_tensor(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)
    return tensor


def _infer_actions_tensor(actions: Any) -> torch.Tensor:
    tensor = torch.as_tensor(actions)
    if tensor.dtype.is_floating_point:
        return tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.int64)


class TransitionDataset:
    def __init__(
        self,
        *,
        obs: Any,
        actions: Any,
        rewards: Any,
        next_obs: Any,
        dones: Any,
    ) -> None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).cpu()
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).cpu()
        actions_tensor = _infer_actions_tensor(actions).cpu()
        rewards_tensor = _to_1d_float_tensor(rewards).cpu()
        dones_tensor = _to_1d_float_tensor(dones).cpu()

        size = int(obs_tensor.shape[0])
        if int(next_obs_tensor.shape[0]) != size:
            raise ValueError("all fields must have the same length")
        if int(actions_tensor.shape[0]) != size:
            raise ValueError("all fields must have the same length")
        if int(rewards_tensor.shape[0]) != size:
            raise ValueError("all fields must have the same length")
        if int(dones_tensor.shape[0]) != size:
            raise ValueError("all fields must have the same length")

        self._size = size
        self.obs = obs_tensor
        self.actions = actions_tensor
        self.rewards = rewards_tensor
        self.next_obs = next_obs_tensor
        self.dones = dones_tensor

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty dataset")

        indices = torch.randint(0, self._size, (int(batch_size),), device=self.obs.device)
        obs = self.obs.index_select(0, indices)
        actions = self.actions.index_select(0, indices)
        rewards = self.rewards.index_select(0, indices)
        next_obs = self.next_obs.index_select(0, indices)
        dones = self.dones.index_select(0, indices)

        return {
            "obs": obs.to(device=device),
            "actions": actions.to(device=device),
            "rewards": rewards.to(device=device),
            "next_obs": next_obs.to(device=device),
            "dones": dones.to(device=device),
        }

