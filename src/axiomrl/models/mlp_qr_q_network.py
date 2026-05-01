from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.mlp_q_network import _build_mlp


class MLPQRQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        num_quantiles: int = 51,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if num_quantiles < 2:
            raise ValueError(f"num_quantiles must be >= 2, got {num_quantiles}")

        self.action_dim = int(action_dim)
        self.num_quantiles = int(num_quantiles)

        self.network = _build_mlp(
            input_dim=int(obs_dim),
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim * self.num_quantiles,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        quantiles = self.network(obs_tensor)
        return quantiles.view(-1, self.action_dim, self.num_quantiles)

    def q_values(self, obs: object) -> torch.Tensor:
        quantiles = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        return quantiles.mean(dim=-1)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.q_values(obs)
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
