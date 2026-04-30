from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from rl_training.models.mlp_td3 import _build_mlp


class MLPDDPGModel(nn.Module):
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

        self.actor_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )
        self.q_network = _build_mlp(
            input_dim=obs_dim + action_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_net.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q_network.parameters()

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def actor(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.actor_net(obs_tensor)

    def q_values(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=obs_tensor.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(-1)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
        return self.q_network(inputs).squeeze(-1)
