from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch


class RecurrentRolloutBuffer:
    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_shape: Sequence[int],
        hidden_size: int,
        num_layers: int = 1,
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = tuple(obs_shape)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.step = 0

        self.obs = torch.zeros((num_steps, num_envs, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.episode_starts = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.hidden_h = torch.zeros(
            (num_steps, num_layers, num_envs, hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.hidden_c = torch.zeros(
            (num_steps, num_layers, num_envs, hidden_size),
            dtype=torch.float32,
            device=self.device,
        )

    def add(
        self,
        *,
        obs: object,
        actions: object,
        rewards: object,
        dones: object,
        episode_starts: object,
        values: object,
        logprobs: object,
        recurrent_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if self.step >= self.num_steps:
            raise IndexError("recurrent rollout buffer is full")

        hidden_h, hidden_c = recurrent_state
        self.obs[self.step].copy_(torch.as_tensor(obs, device=self.device, dtype=self.obs_dtype))
        self.actions[self.step].copy_(torch.as_tensor(actions, device=self.device, dtype=torch.int64))
        self.rewards[self.step].copy_(torch.as_tensor(rewards, device=self.device, dtype=torch.float32))
        self.dones[self.step].copy_(torch.as_tensor(dones, device=self.device, dtype=torch.float32))
        self.episode_starts[self.step].copy_(torch.as_tensor(episode_starts, device=self.device, dtype=torch.float32))
        self.values[self.step].copy_(torch.as_tensor(values, device=self.device, dtype=torch.float32))
        self.logprobs[self.step].copy_(torch.as_tensor(logprobs, device=self.device, dtype=torch.float32))
        self.hidden_h[self.step].copy_(torch.as_tensor(hidden_h, device=self.device, dtype=torch.float32))
        self.hidden_c[self.step].copy_(torch.as_tensor(hidden_c, device=self.device, dtype=torch.float32))
        self.step += 1

    def compute_returns_and_advantages(
        self,
        *,
        last_values: object,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        next_values = torch.as_tensor(last_values, device=self.device, dtype=torch.float32)
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for step in range(self.num_steps - 1, -1, -1):
            if step < self.num_steps - 1:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.advantages + self.values

    def iter_sequence_minibatches(
        self,
        *,
        sequence_length: int,
        sequences_per_batch: int,
        shuffle: bool,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
        if sequences_per_batch <= 0:
            raise ValueError(f"sequences_per_batch must be > 0, got {sequences_per_batch}")

        chunks = [
            (env_index, start)
            for env_index in range(self.num_envs)
            for start in range(0, self.num_steps, sequence_length)
        ]
        order = torch.randperm(len(chunks)).tolist() if shuffle else list(range(len(chunks)))

        for batch_start in range(0, len(order), sequences_per_batch):
            indices = order[batch_start : batch_start + sequences_per_batch]

            obs_batches: list[torch.Tensor] = []
            action_batches: list[torch.Tensor] = []
            logprob_batches: list[torch.Tensor] = []
            advantage_batches: list[torch.Tensor] = []
            return_batches: list[torch.Tensor] = []
            episode_start_batches: list[torch.Tensor] = []
            mask_batches: list[torch.Tensor] = []
            init_h_batches: list[torch.Tensor] = []
            init_c_batches: list[torch.Tensor] = []

            for index in indices:
                env_index, start = chunks[index]
                end = min(start + sequence_length, self.num_steps)
                actual_length = end - start

                obs_chunk = self._pad_time_slice(self.obs[start:end, env_index], sequence_length)
                action_chunk = self._pad_time_slice(self.actions[start:end, env_index], sequence_length)
                logprob_chunk = self._pad_time_slice(self.logprobs[start:end, env_index], sequence_length)
                advantage_chunk = self._pad_time_slice(self.advantages[start:end, env_index], sequence_length)
                return_chunk = self._pad_time_slice(self.returns[start:end, env_index], sequence_length)
                episode_start_chunk = self._pad_time_slice(self.episode_starts[start:end, env_index], sequence_length)
                mask_chunk = torch.zeros(sequence_length, dtype=torch.float32, device=self.device)
                mask_chunk[:actual_length] = 1.0

                obs_batches.append(obs_chunk)
                action_batches.append(action_chunk)
                logprob_batches.append(logprob_chunk)
                advantage_batches.append(advantage_chunk)
                return_batches.append(return_chunk)
                episode_start_batches.append(episode_start_chunk)
                mask_batches.append(mask_chunk)
                init_h_batches.append(self.hidden_h[start, :, env_index])
                init_c_batches.append(self.hidden_c[start, :, env_index])

            yield {
                "obs": torch.stack(obs_batches, dim=1),
                "actions": torch.stack(action_batches, dim=1),
                "logprobs": torch.stack(logprob_batches, dim=1),
                "advantages": torch.stack(advantage_batches, dim=1),
                "returns": torch.stack(return_batches, dim=1),
                "episode_starts": torch.stack(episode_start_batches, dim=1),
                "mask": torch.stack(mask_batches, dim=1),
                "initial_h": torch.stack(init_h_batches, dim=1),
                "initial_c": torch.stack(init_c_batches, dim=1),
            }

    def _pad_time_slice(self, values: torch.Tensor, sequence_length: int) -> torch.Tensor:
        if values.shape[0] == sequence_length:
            return values

        padded = torch.zeros((sequence_length, *values.shape[1:]), dtype=values.dtype, device=self.device)
        padded[: values.shape[0]] = values
        return padded
