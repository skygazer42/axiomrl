from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from rl_training.models.mlp_td3 import _build_mlp


class MLPD4PGModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        v_min: float,
        v_max: float,
        num_atoms: int = 51,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if num_atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {num_atoms}")
        if v_max <= v_min:
            raise ValueError(f"expected v_max > v_min, got v_min={v_min}, v_max={v_max}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        support = torch.linspace(self.v_min, self.v_max, self.num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        self.actor_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )
        self.critic_net = _build_mlp(
            input_dim=obs_dim + action_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.num_atoms,
            activation=activation,
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_net.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.critic_net.parameters()

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

    def distribution_logits(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
        return self.critic_net(inputs)

    def probabilities(self, obs: object, actions: object) -> torch.Tensor:
        logits = self.distribution_logits(obs, actions)
        return F.softmax(logits, dim=-1)

    def q_values(self, obs: object, actions: object) -> torch.Tensor:
        probs = self.probabilities(obs, actions)
        return (probs * self.support).sum(dim=-1)
