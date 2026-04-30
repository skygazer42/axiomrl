from __future__ import annotations

from typing import Any

import torch

from rl_training.data.offline_dataset import TransitionDataset

_FIELD_LENGTH_ERROR = "all fields must have the same length"


class TrajectoryWindowDataset:
    def __init__(
        self,
        *,
        obs: Any,
        actions: Any,
        returns_to_go: Any,
        timesteps: Any,
        mask: Any,
    ) -> None:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).cpu()
        actions_tensor = torch.as_tensor(actions).cpu()
        returns_to_go_tensor = torch.as_tensor(returns_to_go, dtype=torch.float32).cpu()
        timesteps_tensor = torch.as_tensor(timesteps, dtype=torch.int64).cpu()
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32).cpu()

        size = int(obs_tensor.shape[0])
        if int(actions_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(returns_to_go_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(timesteps_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)
        if int(mask_tensor.shape[0]) != size:
            raise ValueError(_FIELD_LENGTH_ERROR)

        self._size = size
        self.context_length = int(obs_tensor.shape[1])
        self.obs = obs_tensor
        self.actions = actions_tensor
        self.returns_to_go = returns_to_go_tensor
        self.timesteps = timesteps_tensor
        self.mask = mask_tensor

    def __len__(self) -> int:
        return self._size

    @classmethod
    def from_transition_dataset(
        cls,
        dataset: TransitionDataset,
        *,
        context_length: int,
    ) -> TrajectoryWindowDataset:
        context = int(context_length)
        if context < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")
        if len(dataset) == 0:
            raise ValueError("cannot build trajectory windows from an empty dataset")
        if dataset.returns_to_go is None:
            raise ValueError("transition dataset must define returns_to_go before building trajectory windows")

        obs = torch.zeros(
            (len(dataset), context, *dataset.obs.shape[1:]),
            dtype=dataset.obs.dtype,
        )
        actions = torch.zeros(
            (len(dataset), context, *dataset.actions.shape[1:]),
            dtype=dataset.actions.dtype,
        )
        returns_to_go = torch.zeros((len(dataset), context), dtype=torch.float32)
        timesteps = torch.zeros((len(dataset), context), dtype=torch.int64)
        mask = torch.zeros((len(dataset), context), dtype=torch.float32)

        episode_start = 0
        sample_index = 0
        done_flags = dataset.dones.reshape(-1) >= 0.5

        for transition_index in range(len(dataset)):
            is_episode_end = bool(done_flags[transition_index].item()) or transition_index == len(dataset) - 1
            if not is_episode_end:
                continue

            episode_end = transition_index + 1
            episode_obs = dataset.obs[episode_start:episode_end]
            episode_actions = dataset.actions[episode_start:episode_end]
            episode_returns = dataset.returns_to_go[episode_start:episode_end]

            for local_index in range(int(episode_obs.shape[0])):
                valid_length = min(context, local_index + 1)
                source_start = local_index + 1 - valid_length
                dest_start = context - valid_length

                obs[sample_index, dest_start:] = episode_obs[source_start : local_index + 1]
                actions[sample_index, dest_start:] = episode_actions[source_start : local_index + 1]
                returns_to_go[sample_index, dest_start:] = episode_returns[source_start : local_index + 1]
                timesteps[sample_index, dest_start:] = torch.arange(source_start, local_index + 1, dtype=torch.int64)
                mask[sample_index, dest_start:] = 1.0
                sample_index += 1

            episode_start = episode_end

        if sample_index != len(dataset):
            raise RuntimeError("failed to build one trajectory window per transition")

        return cls(
            obs=obs,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            mask=mask,
        )

    def sample(self, batch_size: int, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("cannot sample from an empty dataset")

        indices = torch.randint(0, self._size, (int(batch_size),), device=self.obs.device)
        return {
            "obs": self.obs.index_select(0, indices).to(device=device),
            "actions": self.actions.index_select(0, indices).to(device=device),
            "returns_to_go": self.returns_to_go.index_select(0, indices).to(device=device),
            "timesteps": self.timesteps.index_select(0, indices).to(device=device),
            "mask": self.mask.index_select(0, indices).to(device=device),
        }
