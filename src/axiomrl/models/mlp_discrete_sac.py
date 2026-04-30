from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(slots=True)
class DiscreteSACSample:
    actions: torch.Tensor
    logprobs: torch.Tensor
    action_probs: torch.Tensor
    log_action_probs: torch.Tensor


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


class MLPDiscreteSACModel(nn.Module):
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
        self.actor = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )
        self.q1 = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )
        self.q2 = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def policy(self, obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        logits = self.actor(obs_tensor)
        log_action_probs = torch.log_softmax(logits, dim=-1)
        action_probs = log_action_probs.exp()
        return action_probs, log_action_probs

    def sample_actions(self, obs: object, *, deterministic: bool = False) -> DiscreteSACSample:
        action_probs, log_action_probs = self.policy(obs)
        distribution = Categorical(probs=action_probs)
        actions = action_probs.argmax(dim=-1) if deterministic else distribution.sample()
        selected_logprobs = log_action_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return DiscreteSACSample(
            actions=actions,
            logprobs=selected_logprobs,
            action_probs=action_probs,
            log_action_probs=log_action_probs,
        )

    def q_values(self, obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        return self.q1(obs_tensor), self.q2(obs_tensor)

