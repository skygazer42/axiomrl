from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTION_EPS = 1e-6


@dataclass(slots=True)
class IQLSample:
    actions: torch.Tensor
    logprobs: torch.Tensor
    pre_tanh_actions: torch.Tensor


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
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MLPIQLModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.actor_backbone = _build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=hidden_sizes[-1],
            activation=activation,
        )
        self.actor_mean = nn.Linear(hidden_sizes[-1], self.action_dim)
        self.actor_log_std = nn.Linear(hidden_sizes[-1], self.action_dim)

        critic_input_dim = self.obs_dim + self.action_dim
        self.q1 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )
        self.q2 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )
        self.value_net = _build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_backbone.parameters()
        yield from self.actor_mean.parameters()
        yield from self.actor_log_std.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()

    def value_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.value_net.parameters()

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _prepare_actions(self, actions: object, *, device: torch.device) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        return action_tensor

    def _actor_stats(self, obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        features = self.actor_backbone(obs_tensor)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample_actions(self, obs: object, *, deterministic: bool = False) -> IQLSample:
        mean, log_std = self._actor_stats(obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh_actions = mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh_actions)

        logprobs = distribution.log_prob(pre_tanh_actions)
        logprobs = logprobs - torch.log(1.0 - actions.pow(2) + ACTION_EPS)
        logprobs = logprobs.sum(dim=-1)

        return IQLSample(
            actions=actions,
            logprobs=logprobs,
            pre_tanh_actions=pre_tanh_actions,
        )

    def action_logprobs(self, obs: object, actions: object) -> torch.Tensor:
        mean, log_std = self._actor_stats(obs)
        action_tensor = self._prepare_actions(actions, device=mean.device)
        clipped_actions = action_tensor.clamp(-1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
        pre_tanh_actions = 0.5 * torch.log((1.0 + clipped_actions) / (1.0 - clipped_actions))
        distribution = Normal(mean, log_std.exp())
        logprobs = distribution.log_prob(pre_tanh_actions)
        logprobs = logprobs - torch.log(1.0 - clipped_actions.pow(2) + ACTION_EPS)
        return logprobs.sum(dim=-1)

    def q_values(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def value(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.value_net(obs_tensor).squeeze(-1)
