from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rl_training.models.mlp_td3 import _build_mlp


class MLPNAFModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_tril_entries = action_dim * (action_dim + 1) // 2

        self.actor_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )
        self.value_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )
        self.tril_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.num_tril_entries,
            activation=activation,
        )

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _prepare_actions(self, actions: object, *, device: torch.device) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        if action_tensor.ndim == 1:
            if self.action_dim == 1:
                action_tensor = action_tensor.unsqueeze(-1)
            else:
                action_tensor = action_tensor.unsqueeze(0)
        return action_tensor

    def actor(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.actor_net(obs_tensor)

    def state_values(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.value_net(obs_tensor).squeeze(-1)

    def _precision_matrices(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        tril_values = self.tril_net(obs_tensor)
        batch_size = int(tril_values.shape[0])
        tril = torch.zeros(
            (batch_size, self.action_dim, self.action_dim),
            dtype=tril_values.dtype,
            device=tril_values.device,
        )
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0, device=tril_values.device)
        tril[:, tril_indices[0], tril_indices[1]] = tril_values
        diagonal_indices = torch.arange(self.action_dim, device=tril_values.device)
        tril[:, diagonal_indices, diagonal_indices] = F.softplus(tril[:, diagonal_indices, diagonal_indices]) + 1e-3
        return tril @ tril.transpose(-1, -2)

    def q_values(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        greedy_actions = self.actor_net(obs_tensor)
        state_values = self.value_net(obs_tensor).squeeze(-1)
        precision = self._precision_matrices(obs_tensor)
        diff = (action_tensor - greedy_actions).unsqueeze(-1)
        advantages = -0.5 * torch.matmul(diff.transpose(-1, -2), torch.matmul(precision, diff)).squeeze(-1).squeeze(-1)
        return state_values + advantages
