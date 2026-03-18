from __future__ import annotations

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


class CNNIQNetwork(nn.Module):
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
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNIQNetwork expects 3D channel-first observations, got {self.obs_shape!r}")
        if num_quantiles < 2:
            raise ValueError(f"num_quantiles must be >= 2, got {num_quantiles}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        self.action_dim = int(action_dim)
        self.num_quantiles = int(num_quantiles)
        self.embedding_dim = int(embedding_dim)

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

        self.quantile_layer = nn.Linear(self.embedding_dim, self.feature_dim)
        self.quantile_activation = activation()
        self.head = nn.Linear(self.feature_dim, self.action_dim)

        self.register_buffer(
            "cosine_basis",
            torch.arange(1, self.embedding_dim + 1, dtype=torch.float32),
        )

    def _resolve_taus(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        num_quantiles: int | None,
        taus: torch.Tensor | None,
        random_taus: bool,
    ) -> torch.Tensor:
        if taus is not None:
            tau_tensor = torch.as_tensor(taus, dtype=dtype, device=device)
            if tau_tensor.ndim == 1:
                tau_tensor = tau_tensor.unsqueeze(0).expand(batch_size, -1)
            if tau_tensor.ndim != 2:
                raise ValueError("taus must have shape (batch, num_quantiles) or (num_quantiles,)")
            if int(tau_tensor.shape[0]) != batch_size:
                raise ValueError(f"expected taus batch size {batch_size}, got {int(tau_tensor.shape[0])}")
            return tau_tensor

        quantile_count = int(num_quantiles or self.num_quantiles)
        if random_taus:
            return torch.rand((batch_size, quantile_count), dtype=dtype, device=device)

        tau_steps = (torch.arange(quantile_count, dtype=dtype, device=device) + 0.5) / float(quantile_count)
        return tau_steps.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        obs: torch.Tensor,
        *,
        num_quantiles: int | None = None,
        taus: torch.Tensor | None = None,
        random_taus: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        batch_size = int(obs_tensor.shape[0])
        tau_tensor = self._resolve_taus(
            batch_size=batch_size,
            device=obs_tensor.device,
            dtype=obs_tensor.dtype,
            num_quantiles=num_quantiles,
            taus=taus,
            random_taus=random_taus,
        )

        features = self.feature_extractor(obs_tensor)
        state_features = self.trunk_activation(self.trunk(features))

        cosine_embeddings = torch.cos(
            torch.pi
            * tau_tensor.unsqueeze(-1)
            * self.cosine_basis.to(device=obs_tensor.device, dtype=obs_tensor.dtype),
        )
        quantile_features = self.quantile_activation(self.quantile_layer(cosine_embeddings))

        joint_features = state_features.unsqueeze(1) * quantile_features
        quantile_count = int(tau_tensor.shape[1])
        q_values = self.head(joint_features.reshape(batch_size * quantile_count, self.feature_dim))
        q_values = q_values.view(batch_size, quantile_count, self.action_dim).transpose(1, 2)
        return q_values, tau_tensor

    def q_values(self, obs: object, *, num_quantiles: int | None = None) -> torch.Tensor:
        quantiles, _ = self.forward(
            torch.as_tensor(obs, dtype=torch.float32),
            num_quantiles=num_quantiles,
            random_taus=False,
        )
        return quantiles.mean(dim=-1)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.q_values(obs)
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)

