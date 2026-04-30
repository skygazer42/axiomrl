from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.mlp_noisy_q_network import NoisyLinear


class CNNNoisyQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        sigma_init: float = 0.5,
        features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNNoisyQNetwork expects 3D channel-first observations, got {self.obs_shape!r}")

        self.action_dim = int(action_dim)
        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=int(features_dim))

        layers: list[nn.Module] = []
        last_dim = int(features_dim)
        for hidden_dim in hidden_sizes:
            layers.append(NoisyLinear(last_dim, int(hidden_dim), sigma_init=sigma_init))
            layers.append(activation())
            last_dim = int(hidden_dim)
        layers.append(NoisyLinear(last_dim, self.action_dim, sigma_init=sigma_init))
        self.head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.feature_extractor(obs_tensor)
        return self.head(features)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
