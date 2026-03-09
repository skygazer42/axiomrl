from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _build_mlp(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.network = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return self.network(obs_tensor)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
