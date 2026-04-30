from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

_FIELD_LENGTH_ERROR = "all fields must have the same length"


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


def compute_discounted_returns_to_go(
    rewards: Any,
    dones: Any,
    *,
    gamma: float,
) -> torch.Tensor:
    gamma_value = float(gamma)
    if gamma_value < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    rewards_tensor = _to_1d_float_tensor(rewards).cpu()
    dones_tensor = _to_1d_float_tensor(dones).cpu()
    if rewards_tensor.shape != dones_tensor.shape:
        raise ValueError("rewards and dones must have the same shape")

    returns_to_go = torch.zeros_like(rewards_tensor)
    running_return = torch.zeros((), dtype=torch.float32)
    for index in range(int(rewards_tensor.shape[0]) - 1, -1, -1):
        reward = rewards_tensor[index]
        done = dones_tensor[index]
        if bool(done.item() >= 0.5):
            running_return = reward
        else:
            running_return = reward + gamma_value * running_return
        returns_to_go[index] = running_return
    return returns_to_go


class TransitionDataset:
    def __init__(
        self,
        *,
        obs: Any,
        actions: Any,
        rewards: Any,
        next_obs: Any,
        dones: Any,
        next_actions: Any | None = None,
        returns_to_go: Any | None = None,
    ) -> None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).cpu()
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).cpu()
        actions_tensor = _infer_actions_tensor(actions).cpu()
        rewards_tensor = _to_1d_float_tensor(rewards).cpu()
        dones_tensor = _to_1d_float_tensor(dones).cpu()
        next_actions_tensor = _infer_actions_tensor(next_actions).cpu() if next_actions is not None else None
        returns_to_go_tensor = _to_1d_float_tensor(returns_to_go).cpu() if returns_to_go is not None else None

        size = int(obs_tensor.shape[0])
        if int(next_obs_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(actions_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(rewards_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(dones_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if next_actions_tensor is not None and int(next_actions_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if returns_to_go_tensor is not None and int(returns_to_go_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)

        self._size = size
        self.obs = obs_tensor
        self.actions = actions_tensor
        self.rewards = rewards_tensor
        self.next_obs = next_obs_tensor
        self.dones = dones_tensor
        self.next_actions = next_actions_tensor
        self.returns_to_go = returns_to_go_tensor

    def __len__(self) -> int:
        return self._size

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TransitionDataset:
        dones = payload.get("dones")
        if dones is None:
            terminations = payload.get("terminations", payload.get("terminateds"))
            truncations = payload.get("truncations", payload.get("truncateds"))
            if terminations is not None and truncations is not None:
                dones = torch.logical_or(
                    torch.as_tensor(terminations, dtype=torch.bool),
                    torch.as_tensor(truncations, dtype=torch.bool),
                ).to(dtype=torch.float32)
            elif terminations is not None:
                dones = terminations
            else:
                raise KeyError("transition payload must define 'dones' or 'terminations'")

        return cls(
            obs=payload["obs"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            next_obs=payload["next_obs"],
            dones=dones,
            next_actions=payload.get("next_actions"),
            returns_to_go=payload.get("returns_to_go"),
        )

    def sample(self, batch_size: int, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty dataset")

        indices = torch.randint(0, self._size, (int(batch_size),), device=self.obs.device)
        obs = self.obs.index_select(0, indices)
        actions = self.actions.index_select(0, indices)
        rewards = self.rewards.index_select(0, indices)
        next_obs = self.next_obs.index_select(0, indices)
        dones = self.dones.index_select(0, indices)
        next_actions = self.next_actions.index_select(0, indices) if self.next_actions is not None else None
        returns_to_go = self.returns_to_go.index_select(0, indices) if self.returns_to_go is not None else None

        batch = {
            "obs": obs.to(device=device),
            "actions": actions.to(device=device),
            "rewards": rewards.to(device=device),
            "next_obs": next_obs.to(device=device),
            "dones": dones.to(device=device),
        }
        if next_actions is not None:
            batch["next_actions"] = next_actions.to(device=device)
        if returns_to_go is not None:
            batch["returns_to_go"] = returns_to_go.to(device=device)
        return batch

    def with_reward_transform(
        self,
        *,
        scale: float = 1.0,
        shift: float = 0.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        returns_to_go_gamma: float | None = None,
    ) -> TransitionDataset:
        rewards = self.rewards.clone()
        scale_value = float(scale)
        shift_value = float(shift)
        if not math.isclose(scale_value, 1.0):
            rewards = rewards * scale_value
        if not math.isclose(shift_value, 0.0):
            rewards = rewards + shift_value
        if clip_min is not None or clip_max is not None:
            lower = float("-inf") if clip_min is None else float(clip_min)
            upper = float("inf") if clip_max is None else float(clip_max)
            rewards = rewards.clamp(min=lower, max=upper)

        returns_to_go = None
        if returns_to_go_gamma is not None:
            returns_to_go = compute_discounted_returns_to_go(
                rewards,
                self.dones,
                gamma=float(returns_to_go_gamma),
            )
        elif (
            self.returns_to_go is not None
            and math.isclose(scale_value, 1.0)
            and math.isclose(shift_value, 0.0)
            and clip_min is None
            and clip_max is None
        ):
            returns_to_go = self.returns_to_go.clone()

        return TransitionDataset(
            obs=self.obs.clone(),
            actions=self.actions.clone(),
            rewards=rewards,
            next_obs=self.next_obs.clone(),
            dones=self.dones.clone(),
            next_actions=self.next_actions.clone() if self.next_actions is not None else None,
            returns_to_go=returns_to_go,
        )

    def with_discounted_returns_to_go(self, *, gamma: float) -> TransitionDataset:
        return TransitionDataset(
            obs=self.obs.clone(),
            actions=self.actions.clone(),
            rewards=self.rewards.clone(),
            next_obs=self.next_obs.clone(),
            dones=self.dones.clone(),
            next_actions=self.next_actions.clone() if self.next_actions is not None else None,
            returns_to_go=compute_discounted_returns_to_go(self.rewards, self.dones, gamma=float(gamma)),
        )
