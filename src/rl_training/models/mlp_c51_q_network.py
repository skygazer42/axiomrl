from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rl_training.models.mlp_q_network import _build_mlp


class MLPC51QNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        v_min: float,
        v_max: float,
        num_atoms: int = 51,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if num_atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {num_atoms}")
        if v_max <= v_min:
            raise ValueError(f"expected v_max > v_min, got v_min={v_min}, v_max={v_max}")

        self.action_dim = int(action_dim)
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        support = torch.linspace(self.v_min, self.v_max, self.num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        self.network = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim * self.num_atoms,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        logits = self.network(obs_tensor)
        return logits.view(-1, self.action_dim, self.num_atoms)

    def probabilities(self, obs: object) -> torch.Tensor:
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        return F.softmax(logits, dim=-1)

    def q_values(self, obs: object) -> torch.Tensor:
        probs = self.probabilities(obs)
        return (probs * self.support).sum(dim=-1)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.q_values(obs)
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)

