from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from axiomrl.models.cnn.nature import NatureCNN


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
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def _prepare_obs(obs: object) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor


def _prepare_actions(actions: object, *, device: torch.device) -> torch.Tensor:
    action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
    if action_tensor.ndim == 0:
        action_tensor = action_tensor.unsqueeze(0)
    return action_tensor


class MLPGAILDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.network = _build_mlp(
            input_dim=self.obs_dim + self.action_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def forward(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = _prepare_obs(obs)
        action_tensor = _prepare_actions(actions, device=obs_tensor.device)
        one_hot = F.one_hot(action_tensor, num_classes=self.action_dim).to(dtype=torch.float32)
        inputs = torch.cat([obs_tensor, one_hot], dim=-1)
        return self.network(inputs).squeeze(-1)


class CNNGAILDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = int(action_dim)
        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.network = _build_mlp(
            input_dim=int(features_dim) + self.action_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def forward(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = _prepare_actions(actions, device=obs_tensor.device)
        features = self.feature_extractor(obs_tensor)
        one_hot = F.one_hot(action_tensor, num_classes=self.action_dim).to(dtype=torch.float32)
        inputs = torch.cat([features, one_hot], dim=-1)
        return self.network(inputs).squeeze(-1)
