from collections.abc import Iterator, Sequence

import torch


class RolloutBuffer:
    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        device: str | torch.device = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype or (torch.int64 if not self.action_shape else torch.float32)
        self.step = 0

        self.obs = torch.zeros((num_steps, num_envs, *self.obs_shape), dtype=self.obs_dtype, device=self.device)
        self.actions = torch.zeros(
            (num_steps, num_envs, *self.action_shape),
            dtype=self.action_dtype,
            device=self.device,
        )
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.step = 0
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.logprobs.zero_()
        self.advantages.zero_()
        self.returns.zero_()

    def add(
        self,
        *,
        obs: object,
        actions: object,
        rewards: object,
        dones: object,
        values: object,
        logprobs: object,
    ) -> None:
        if self.step >= self.num_steps:
            raise IndexError("rollout buffer is full")

        self.obs[self.step].copy_(torch.as_tensor(obs, device=self.device, dtype=self.obs_dtype))
        self.actions[self.step].copy_(torch.as_tensor(actions, device=self.device, dtype=self.action_dtype))
        self.rewards[self.step].copy_(torch.as_tensor(rewards, device=self.device, dtype=torch.float32))
        self.dones[self.step].copy_(torch.as_tensor(dones, device=self.device, dtype=torch.float32))
        self.values[self.step].copy_(torch.as_tensor(values, device=self.device, dtype=torch.float32))
        self.logprobs[self.step].copy_(torch.as_tensor(logprobs, device=self.device, dtype=torch.float32))
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

    def iter_minibatches(
        self,
        *,
        minibatch_size: int,
        shuffle: bool,
    ) -> Iterator[dict[str, torch.Tensor]]:
        total_items = self.num_steps * self.num_envs
        indices = (
            torch.randperm(total_items, device=self.device)
            if shuffle
            else torch.arange(total_items, device=self.device)
        )

        flattened = {
            "obs": self.obs.reshape(total_items, *self.obs_shape),
            "actions": self.actions.reshape(total_items, *self.action_shape)
            if self.action_shape
            else self.actions.reshape(total_items),
            "rewards": self.rewards.reshape(total_items),
            "dones": self.dones.reshape(total_items),
            "values": self.values.reshape(total_items),
            "logprobs": self.logprobs.reshape(total_items),
            "advantages": self.advantages.reshape(total_items),
            "returns": self.returns.reshape(total_items),
        }

        for start in range(0, total_items, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            yield {name: tensor[batch_indices] for name, tensor in flattened.items()}
