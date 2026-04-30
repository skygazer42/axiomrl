from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from rl_training.models.cnn.nature import NatureCNN
from rl_training.models.dreamer import DreamerModel
from rl_training.policies.base import PolicyOutput


class PODreamerModel(DreamerModel):
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
        memory_dim: int = 64,
        memory_hidden_size: int = 128,
        memory_mix: float = 0.35,
    ) -> None:
        if int(memory_dim) < 1:
            raise ValueError(f"memory_dim must be >= 1, got {memory_dim}")
        if int(memory_hidden_size) < 1:
            raise ValueError(f"memory_hidden_size must be >= 1, got {memory_hidden_size}")
        if not 0.0 <= float(memory_mix) <= 1.0:
            raise ValueError(f"memory_mix must be in [0, 1], got {memory_mix}")

        self.memory_dim = int(memory_dim)
        self.memory_hidden_size = int(memory_hidden_size)
        self.memory_mix = float(memory_mix)
        super().__init__(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=features_dim,
            action_embed_dim=action_embed_dim,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            reward_hidden_sizes=reward_hidden_sizes,
        )

        self.memory_encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=self.memory_dim)
        self.memory_projection = nn.Linear(self.memory_dim, self.features_dim)
        self.memory_gate = nn.Sequential(
            nn.Linear(self.features_dim * 2, self.memory_hidden_size),
            nn.ReLU(),
            nn.Linear(self.memory_hidden_size, self.features_dim),
            nn.Sigmoid(),
        )
        self.memory_predictor = self._build_mlp(self.features_dim, (self.memory_hidden_size,), self.memory_dim)

    def encode_memory(self, obs: torch.Tensor) -> torch.Tensor:
        return self.memory_encoder(obs)

    def predict_memory(self, features: torch.Tensor) -> torch.Tensor:
        return self.memory_predictor(features)

    def fuse_with_memory(
        self,
        features: torch.Tensor,
        memory_features: torch.Tensor,
        *,
        detach_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_memory = self.memory_projection(memory_features)
        if detach_memory:
            projected_memory = projected_memory.detach()
        gate = self.memory_gate(torch.cat([features, projected_memory], dim=-1))
        fused_features = features + self.memory_mix * gate * projected_memory
        return fused_features, gate

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        del state
        device = next(self.parameters()).device
        obs_tensor = self._prepare_obs(obs, device=device)
        features = self.encode(obs_tensor)
        memory_features = self.encode_memory(obs_tensor)
        fused_features, _ = self.fuse_with_memory(features, memory_features, detach_memory=True)
        logits = self.actor_logits(fused_features)
        distribution = Categorical(logits=logits)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value(fused_features)
        return PolicyOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            entropy=entropy,
            state=None,
        )

    def parameters_world_model(self) -> Iterator[nn.Parameter]:
        yield from self.encoder.parameters()
        yield from self.action_embedding.parameters()
        yield from self.dynamics.parameters()
        yield from self.decoder_fc.parameters()
        yield from self.decoder.parameters()
        yield from self.reward_head.parameters()
        yield from self.memory_encoder.parameters()
        yield from self.memory_projection.parameters()
        yield from self.memory_gate.parameters()
        yield from self.memory_predictor.parameters()
