from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.mlp_td3 import _build_mlp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTION_EPS = 1e-6


@dataclass(slots=True)
class CURLSample:
    actions: torch.Tensor
    logprobs: torch.Tensor
    pre_tanh_actions: torch.Tensor


class CNNCURLModel(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        features_dim: int = 256,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
        projection_dim: int = 128,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = action_dim
        self.features_dim = features_dim
        self.projection_dim = projection_dim

        self.actor_encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.critic_encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.actor_backbone = _build_mlp(
            input_dim=features_dim,
            hidden_sizes=actor_hidden_sizes,
            output_dim=actor_hidden_sizes[-1],
            activation=activation,
        )
        self.actor_mean = nn.Linear(actor_hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Linear(actor_hidden_sizes[-1], action_dim)

        critic_input_dim = features_dim + action_dim
        self.q1 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_sizes=critic_hidden_sizes,
            output_dim=1,
            activation=activation,
        )
        self.q2 = _build_mlp(
            input_dim=critic_input_dim,
            hidden_sizes=critic_hidden_sizes,
            output_dim=1,
            activation=activation,
        )
        self.curl_projection = nn.Sequential(
            nn.Linear(features_dim, projection_dim),
            activation(),
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_encoder.parameters()
        yield from self.actor_backbone.parameters()
        yield from self.actor_mean.parameters()
        yield from self.actor_log_std.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.critic_encoder.parameters()
        yield from self.q1.parameters()
        yield from self.q2.parameters()
        yield from self.curl_projection.parameters()

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
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

    def _actor_stats(self, obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        features = self.actor_encoder(obs_tensor)
        actor_features = self.actor_backbone(features)
        mean = self.actor_mean(actor_features)
        log_std = self.actor_log_std(actor_features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def actor(self, obs: object) -> torch.Tensor:
        mean, _ = self._actor_stats(obs)
        return torch.tanh(mean)

    def sample_actions(self, obs: object, *, deterministic: bool = False) -> CURLSample:
        mean, log_std = self._actor_stats(obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh_actions = mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh_actions)

        logprobs = distribution.log_prob(pre_tanh_actions)
        logprobs = logprobs - torch.log(1.0 - actions.pow(2) + ACTION_EPS)
        logprobs = logprobs.sum(dim=-1)

        return CURLSample(
            actions=actions,
            logprobs=logprobs,
            pre_tanh_actions=pre_tanh_actions,
        )

    def q_values(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        features = self.critic_encoder(obs_tensor)
        critic_inputs = torch.cat([features, action_tensor], dim=-1)
        return self.q1(critic_inputs).squeeze(-1), self.q2(critic_inputs).squeeze(-1)

    def curl_embeddings(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        features = self.critic_encoder(obs_tensor)
        projected = self.curl_projection(features)
        return F.normalize(projected, dim=-1)
