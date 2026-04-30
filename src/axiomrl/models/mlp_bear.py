from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn
from torch.distributions import Normal

from axiomrl.models.mlp_sac import ACTION_EPS, LOG_STD_MAX, LOG_STD_MIN, SACSample
from axiomrl.models.mlp_td3 import _build_mlp
from axiomrl.policies.base import PolicyOutput

LATENT_LOG_STD_MIN = -4.0
LATENT_LOG_STD_MAX = 4.0
LATENT_CLIP = 0.5


class MLPBEARModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if int(obs_dim) < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        if int(action_dim) < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)

        self.actor_backbone = _build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=hidden_sizes[-1],
            activation=activation,
        )
        self.actor_mean = nn.Linear(hidden_sizes[-1], self.action_dim)
        self.actor_log_std = nn.Linear(hidden_sizes[-1], self.action_dim)

        behavior_input_dim = self.obs_dim + self.action_dim
        self.behavior_encoder = _build_mlp(
            input_dim=behavior_input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=hidden_sizes[-1],
            activation=activation,
        )
        self.behavior_latent_mean = nn.Linear(hidden_sizes[-1], self.latent_dim)
        self.behavior_latent_log_std = nn.Linear(hidden_sizes[-1], self.latent_dim)
        self.behavior_decoder = _build_mlp(
            input_dim=self.obs_dim + self.latent_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )

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

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_backbone.parameters()
        yield from self.actor_mean.parameters()
        yield from self.actor_log_std.parameters()

    def behavior_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.behavior_encoder.parameters()
        yield from self.behavior_latent_mean.parameters()
        yield from self.behavior_latent_log_std.parameters()
        yield from self.behavior_decoder.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()

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

    def _repeat_observations(self, obs_tensor: torch.Tensor, count: int) -> torch.Tensor:
        return obs_tensor.unsqueeze(1).expand(-1, count, -1).reshape(obs_tensor.shape[0] * count, obs_tensor.shape[1])

    def _actor_stats(self, obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        features = self.actor_backbone(obs_tensor)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample_actions(self, obs: object, *, deterministic: bool = False) -> SACSample:
        mean, log_std = self._actor_stats(obs)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh_actions = mean if deterministic else distribution.rsample()
        actions = torch.tanh(pre_tanh_actions)

        logprobs = distribution.log_prob(pre_tanh_actions)
        logprobs = logprobs - torch.log(1.0 - actions.pow(2) + ACTION_EPS)
        logprobs = logprobs.sum(dim=-1)

        return SACSample(
            actions=actions,
            logprobs=logprobs,
            pre_tanh_actions=pre_tanh_actions,
        )

    def encode_behavior(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        features = self.behavior_encoder(torch.cat([obs_tensor, action_tensor], dim=-1))
        mean = self.behavior_latent_mean(features)
        log_std = self.behavior_latent_log_std(features).clamp(LATENT_LOG_STD_MIN, LATENT_LOG_STD_MAX)
        return mean, log_std

    def decode_behavior(
        self,
        obs: object,
        *,
        latent: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        if latent is None:
            if deterministic:
                latent_tensor = torch.zeros(
                    (obs_tensor.shape[0], self.latent_dim),
                    dtype=torch.float32,
                    device=obs_tensor.device,
                )
            else:
                latent_tensor = torch.randn(
                    (obs_tensor.shape[0], self.latent_dim),
                    dtype=torch.float32,
                    device=obs_tensor.device,
                ).clamp(-LATENT_CLIP, LATENT_CLIP)
        else:
            latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=obs_tensor.device)
            if latent_tensor.ndim == 1:
                latent_tensor = latent_tensor.unsqueeze(0)
            latent_tensor = latent_tensor.clamp(-LATENT_CLIP, LATENT_CLIP)
        return self.behavior_decoder(torch.cat([obs_tensor, latent_tensor], dim=-1))

    def reconstruct_behavior(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.encode_behavior(obs, actions)
        std = log_std.exp()
        latent = mean + std * torch.randn_like(std)
        reconstructed_actions = self.decode_behavior(obs, latent=latent)
        return reconstructed_actions, mean, log_std

    def sample_behavior_actions(
        self,
        obs: object,
        *,
        num_action_samples: int,
        deterministic: bool = False,
    ) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        sample_count = int(num_action_samples)
        repeated_obs = self._repeat_observations(obs_tensor, sample_count)
        decoded_actions = self.decode_behavior(repeated_obs, deterministic=deterministic)
        return decoded_actions.reshape(obs_tensor.shape[0], sample_count, self.action_dim)

    def q_values(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = True,
    ) -> PolicyOutput:
        del state
        sample = self.sample_actions(obs, deterministic=deterministic)
        return PolicyOutput(
            actions=sample.actions,
            logprobs=sample.logprobs,
            values=None,
            entropy=None,
            state=None,
        )
