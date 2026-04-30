from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn

from axiomrl.models.dreamer import DreamerModel


class EADreamModel(DreamerModel):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        features_dim: int = 128,
        action_embed_dim: int = 32,
        actor_hidden_sizes: Sequence[int] = (256, 256),
        critic_hidden_sizes: Sequence[int] = (256, 256),
        reward_hidden_sizes: Sequence[int] = (256, 256),
        event_hidden_sizes: Sequence[int] = (128,),
        event_scale: float = 1.0,
    ) -> None:
        if float(event_scale) < 0.0:
            raise ValueError(f"event_scale must be >= 0, got {event_scale}")

        self.event_scale = float(event_scale)
        super().__init__(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=features_dim,
            action_embed_dim=action_embed_dim,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            reward_hidden_sizes=reward_hidden_sizes,
        )
        self.event_head = self._build_mlp(self.features_dim, event_hidden_sizes, 1)

    def event_probability(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.event_head(features)).squeeze(-1)

    def event_aware_features(self, features: torch.Tensor, *, detach_event: bool = True) -> torch.Tensor:
        event_probabilities = self.event_probability(features)
        if detach_event:
            event_probabilities = event_probabilities.detach()
        return features * (1.0 + self.event_scale * event_probabilities.unsqueeze(-1))

    def actor_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.event_aware_features(features))

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.event_aware_features(features)).squeeze(-1)

    def parameters_world_model(self) -> Iterator[nn.Parameter]:
        yield from self.encoder.parameters()
        yield from self.action_embedding.parameters()
        yield from self.dynamics.parameters()
        yield from self.decoder_fc.parameters()
        yield from self.decoder.parameters()
        yield from self.reward_head.parameters()
        yield from self.event_head.parameters()
