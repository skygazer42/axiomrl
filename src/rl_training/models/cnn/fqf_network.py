from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from rl_training.models.cnn.nature import NatureCNN


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


@dataclass(frozen=True, slots=True)
class FQFNetworkOutput:
    quantile_hats: torch.Tensor
    taus: torch.Tensor
    tau_hats: torch.Tensor
    quantiles_tau: torch.Tensor
    entropies: torch.Tensor


class CNNFQFNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        num_quantiles: int = 32,
        hidden_sizes: Sequence[int] = (512,),
        embedding_dim: int = 64,
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
        entropy_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNFQFNetwork expects 3D channel-first observations, got {self.obs_shape!r}")
        if num_quantiles < 2:
            raise ValueError(f"num_quantiles must be >= 2, got {num_quantiles}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        if entropy_eps <= 0:
            raise ValueError(f"entropy_eps must be > 0, got {entropy_eps}")

        self.action_dim = int(action_dim)
        self.num_quantiles = int(num_quantiles)
        self.embedding_dim = int(embedding_dim)
        self.entropy_eps = float(entropy_eps)

        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=int(features_dim))

        if hidden_sizes:
            trunk_hidden_sizes = tuple(int(dim) for dim in hidden_sizes[:-1])
            self.feature_dim = int(hidden_sizes[-1])
            self.trunk = _build_head(
                input_dim=int(features_dim),
                hidden_sizes=trunk_hidden_sizes,
                output_dim=self.feature_dim,
                activation=activation,
            )
            self.trunk_activation: nn.Module = activation()
        else:
            self.feature_dim = int(features_dim)
            self.trunk = nn.Identity()
            self.trunk_activation = nn.Identity()

        self.fraction_head = nn.Linear(self.feature_dim, self.num_quantiles)

        self.quantile_layer = nn.Linear(self.embedding_dim, self.feature_dim)
        self.quantile_activation = activation()
        self.head = nn.Linear(self.feature_dim, self.action_dim)

        self.register_buffer(
            "cosine_basis",
            torch.arange(1, self.embedding_dim + 1, dtype=torch.float32),
        )

    def fraction_parameters(self) -> list[nn.Parameter]:
        return list(self.fraction_head.parameters())

    def quantile_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(self.feature_extractor.parameters())
        params.extend(self.trunk.parameters())
        params.extend(self.quantile_layer.parameters())
        params.extend(self.head.parameters())
        return params

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _state_features(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        features = self.feature_extractor(obs_tensor)
        return self.trunk_activation(self.trunk(features))

    def _propose_fractions(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.fraction_head(features)
        probs = torch.softmax(logits, dim=-1)

        taus = torch.cat(
            [
                torch.zeros((int(features.shape[0]), 1), dtype=features.dtype, device=features.device),
                probs.cumsum(dim=-1),
            ],
            dim=-1,
        )
        tau_hats = 0.5 * (taus[:, :-1] + taus[:, 1:])
        entropies = -(probs * (probs + self.entropy_eps).log()).sum(dim=-1)
        return taus, tau_hats, entropies

    def quantiles(self, obs: object, taus: torch.Tensor) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        features = self._state_features(obs_tensor)

        tau_tensor = torch.as_tensor(taus, dtype=features.dtype, device=features.device)
        if tau_tensor.ndim == 1:
            tau_tensor = tau_tensor.unsqueeze(0).expand(int(features.shape[0]), -1)
        if tau_tensor.ndim != 2:
            raise ValueError("taus must have shape (batch, num_quantiles) or (num_quantiles,)")
        if int(tau_tensor.shape[0]) != int(features.shape[0]):
            raise ValueError(f"expected taus batch size {int(features.shape[0])}, got {int(tau_tensor.shape[0])}")

        batch_size = int(features.shape[0])
        quantile_count = int(tau_tensor.shape[1])

        cosine_embeddings = torch.cos(
            torch.pi * tau_tensor.unsqueeze(-1) * self.cosine_basis.to(device=features.device, dtype=features.dtype)
        )
        quantile_features = self.quantile_activation(self.quantile_layer(cosine_embeddings))
        joint_features = features.unsqueeze(1) * quantile_features

        q_values = self.head(joint_features.reshape(batch_size * quantile_count, self.feature_dim))
        q_values = q_values.view(batch_size, quantile_count, self.action_dim).transpose(1, 2)
        return q_values

    def forward(self, obs: torch.Tensor, *, detach_quantiles_tau: bool = False) -> FQFNetworkOutput:
        obs_tensor = self._prepare_obs(obs)
        features = self._state_features(obs_tensor)
        taus, tau_hats, entropies = self._propose_fractions(features)
        quantile_hats = self.quantiles(obs_tensor, tau_hats)
        quantiles_tau = self.quantiles(obs_tensor, taus[:, 1:-1])
        if detach_quantiles_tau:
            quantiles_tau = quantiles_tau.detach()
        return FQFNetworkOutput(
            quantile_hats=quantile_hats,
            taus=taus,
            tau_hats=tau_hats,
            quantiles_tau=quantiles_tau,
            entropies=entropies,
        )

    def q_values(self, obs: object) -> torch.Tensor:
        out = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        weights = (out.taus[:, 1:] - out.taus[:, :-1]).unsqueeze(1)
        return (weights * out.quantile_hats).sum(dim=-1)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.q_values(obs)
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)

