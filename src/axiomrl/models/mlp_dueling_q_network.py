from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.mlp_q_network import _build_mlp


class MLPDuelingQNetwork(nn.Module):
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

        if hidden_sizes:
            trunk_hidden_sizes = tuple(hidden_sizes[:-1])
            head_input_dim = int(hidden_sizes[-1])
            self.trunk = _build_mlp(
                input_dim=obs_dim,
                hidden_sizes=trunk_hidden_sizes,
                output_dim=head_input_dim,
                activation=activation,
            )
            self.trunk_activation = activation()
        else:
            head_input_dim = obs_dim
            self.trunk = nn.Identity()
            self.trunk_activation = nn.Identity()

        self.value_head = nn.Linear(head_input_dim, 1)
        self.advantage_head = nn.Linear(head_input_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.trunk_activation(self.trunk(obs_tensor))
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        centered_advantages = advantages - advantages.mean(dim=-1, keepdim=True)
        return values + centered_advantages

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
