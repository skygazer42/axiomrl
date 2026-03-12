from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from rl_training.models.cnn.nature import NatureCNN
from rl_training.models.mlp_td3 import _build_mlp


@dataclass(slots=True)
class DrQv2Sample:
    actions: torch.Tensor
    noise: torch.Tensor


class CNNDrQv2Model(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        features_dim: int = 512,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = action_dim
        self.features_dim = features_dim

        self.actor_encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.critic_encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.actor_net = _build_mlp(
            input_dim=features_dim,
            hidden_sizes=actor_hidden_sizes,
            output_dim=action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )
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

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.actor_encoder.parameters()
        yield from self.actor_net.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.critic_encoder.parameters()
        yield from self.q1.parameters()
        yield from self.q2.parameters()

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

    def actor(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        features = self.actor_encoder(obs_tensor)
        return self.actor_net(features)

    def sample_actions(
        self,
        obs: object,
        *,
        std: float = 0.1,
        clip: float | None = 0.3,
        deterministic: bool = False,
    ) -> DrQv2Sample:
        actions = self.actor(obs)
        if deterministic or std <= 0:
            noise = torch.zeros_like(actions)
            sampled_actions = actions
        else:
            noise = torch.randn_like(actions) * float(std)
            if clip is not None and clip > 0:
                noise = noise.clamp(-clip, clip)
            sampled_actions = (actions + noise).clamp(-1.0, 1.0)
        return DrQv2Sample(actions=sampled_actions, noise=noise)

    def q_values(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        features = self.critic_encoder(obs_tensor)
        critic_inputs = torch.cat([features, action_tensor], dim=-1)
        return self.q1(critic_inputs).squeeze(-1), self.q2(critic_inputs).squeeze(-1)
