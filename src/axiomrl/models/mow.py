from collections.abc import Iterator, Sequence

import torch
from torch import nn

from axiomrl.models.dreamer import DreamerModel


class MoWModel(DreamerModel):
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
        num_experts: int = 4,
        gating_hidden_size: int = 128,
    ) -> None:
        if int(num_experts) < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if int(gating_hidden_size) < 1:
            raise ValueError(f"gating_hidden_size must be >= 1, got {gating_hidden_size}")

        self.num_experts = int(num_experts)
        self.gating_hidden_size = int(gating_hidden_size)
        super().__init__(
            obs_shape=obs_shape,
            action_dim=action_dim,
            features_dim=features_dim,
            action_embed_dim=action_embed_dim,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            reward_hidden_sizes=reward_hidden_sizes,
        )

        self.world_model_gate = self._build_gate(self.features_dim + self.action_embed_dim)
        self.reward_gate = self._build_gate(self.features_dim)
        self.actor_gate = self._build_gate(self.features_dim)
        self.critic_gate = self._build_gate(self.features_dim)

        self.dynamics = nn.ModuleList(
            nn.GRUCell(self.action_embed_dim, self.features_dim) for _ in range(self.num_experts)
        )
        self.reward_head = nn.ModuleList(
            self._build_mlp(self.features_dim, reward_hidden_sizes, 1) for _ in range(self.num_experts)
        )
        self.actor_head = nn.ModuleList(
            self._build_mlp(self.features_dim, actor_hidden_sizes, self.action_dim) for _ in range(self.num_experts)
        )
        self.critic_head = nn.ModuleList(
            self._build_mlp(self.features_dim, critic_hidden_sizes, 1) for _ in range(self.num_experts)
        )

    def _build_gate(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(input_dim), self.gating_hidden_size),
            nn.ReLU(),
            nn.Linear(self.gating_hidden_size, self.num_experts),
        )

    @staticmethod
    def gate_entropy(gate_probs: torch.Tensor) -> torch.Tensor:
        safe_gate_probs = gate_probs.clamp_min(1e-8)
        return -(safe_gate_probs * safe_gate_probs.log()).sum(dim=-1)

    @staticmethod
    def _mix_expert_outputs(outputs: list[torch.Tensor], gate_probs: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(outputs, dim=1)
        weights = gate_probs
        for _ in range(stacked.ndim - weights.ndim):
            weights = weights.unsqueeze(-1)
        return (stacked * weights).sum(dim=1)

    def _compute_gate_probs(self, gate_network: nn.Module, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(gate_network(features), dim=-1)

    def dynamics_step_with_gates(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=features.device)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        action_embed = self.action_embedding(action_tensor)
        gate_input = torch.cat([features, action_embed], dim=-1)
        gate_probs = self._compute_gate_probs(self.world_model_gate, gate_input)
        next_features = self._mix_expert_outputs(
            [expert(action_embed, features) for expert in self.dynamics],
            gate_probs,
        )
        return next_features, gate_probs

    def dynamics_step(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        next_features, _ = self.dynamics_step_with_gates(features, actions)
        return next_features

    def predict_reward_with_gates(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_probs = self._compute_gate_probs(self.reward_gate, features)
        rewards = self._mix_expert_outputs([head(features) for head in self.reward_head], gate_probs)
        return rewards.squeeze(-1), gate_probs

    def predict_reward(self, features: torch.Tensor) -> torch.Tensor:
        rewards, _ = self.predict_reward_with_gates(features)
        return rewards

    def actor_logits_with_gates(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_probs = self._compute_gate_probs(self.actor_gate, features)
        logits = self._mix_expert_outputs([head(features) for head in self.actor_head], gate_probs)
        return logits, gate_probs

    def actor_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self.actor_logits_with_gates(features)
        return logits

    def value_with_gates(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_probs = self._compute_gate_probs(self.critic_gate, features)
        values = self._mix_expert_outputs([head(features) for head in self.critic_head], gate_probs)
        return values.squeeze(-1), gate_probs

    def value(self, features: torch.Tensor) -> torch.Tensor:
        values, _ = self.value_with_gates(features)
        return values

    def parameters_world_model(self) -> Iterator[nn.Parameter]:
        yield from self.encoder.parameters()
        yield from self.action_embedding.parameters()
        yield from self.world_model_gate.parameters()
        yield from self.dynamics.parameters()
        yield from self.decoder_fc.parameters()
        yield from self.decoder.parameters()
        yield from self.reward_gate.parameters()
        yield from self.reward_head.parameters()

    def parameters_actor(self) -> Iterator[nn.Parameter]:
        yield from self.actor_gate.parameters()
        yield from self.actor_head.parameters()

    def parameters_critic(self) -> Iterator[nn.Parameter]:
        yield from self.critic_gate.parameters()
        yield from self.critic_head.parameters()
