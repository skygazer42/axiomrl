from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN


def _build_head(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class CNNQRQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        num_quantiles: int = 51,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNQRQNetwork expects 3D channel-first observations, got {self.obs_shape!r}")
        if num_quantiles < 2:
            raise ValueError(f"num_quantiles must be >= 2, got {num_quantiles}")

        self.action_dim = int(action_dim)
        self.num_quantiles = int(num_quantiles)

        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=int(features_dim))
        self.head = _build_head(
            input_dim=int(features_dim),
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim * self.num_quantiles,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.feature_extractor(obs_tensor)
        quantiles = self.head(features)
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
