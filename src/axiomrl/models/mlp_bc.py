from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from axiomrl.models.mlp_td3 import _build_mlp
from axiomrl.policies.base import PolicyOutput


class MLPBCModel(nn.Module):
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

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_net.parameters()

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def actor(self, obs: object) -> torch.Tensor:
        return self.actor_net(self._prepare_obs(obs))

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = True,
    ) -> PolicyOutput:
        del state, deterministic
        return PolicyOutput(
            actions=self.actor(obs),
            logprobs=None,
            values=None,
            entropy=None,
            state=None,
        )
