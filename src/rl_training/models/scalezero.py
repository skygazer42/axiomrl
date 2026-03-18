from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rl_training.models.muzero import MuZeroInitialOutput, MuZeroModel, MuZeroRecurrentOutput, _build_mlp


class ScaleZeroModel(MuZeroModel):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        latent_dim: int = 256,
        action_embed_dim: int = 64,
        representation_features_dim: int | None = None,
        dynamics_hidden_sizes: Sequence[int] = (256,),
        prediction_hidden_sizes: Sequence[int] = (256,),
        activation: type[nn.Module] = nn.ReLU,
        normalize_latent: bool = True,
        num_experts: int = 4,
        gating_hidden_size: int = 128,
    ) -> None:
        if int(num_experts) < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if int(gating_hidden_size) < 1:
            raise ValueError(f"gating_hidden_size must be >= 1, got {gating_hidden_size}")

        self.num_experts = int(num_experts)
        self.gating_hidden_size = int(gating_hidden_size)
        self._activation = activation
        super().__init__(
            obs_shape=obs_shape,
            action_dim=action_dim,
            latent_dim=latent_dim,
            action_embed_dim=action_embed_dim,
            representation_features_dim=representation_features_dim,
            dynamics_hidden_sizes=dynamics_hidden_sizes,
            prediction_hidden_sizes=prediction_hidden_sizes,
            activation=activation,
            normalize_latent=normalize_latent,
        )

        self.dynamics_gate = self._build_gate(self.latent_dim + self.action_embed_dim)
        self.dynamics = nn.ModuleList(
            _build_mlp(
                input_dim=self.latent_dim + self.action_embed_dim,
                hidden_sizes=dynamics_hidden_sizes,
                output_dim=self.latent_dim,
                activation=self._activation,
            )
            for _ in range(self.num_experts)
        )

        self.prediction_gate = self._build_gate(self.latent_dim)
        self.prediction_trunk = nn.ModuleList(
            _build_mlp(
                input_dim=self.latent_dim,
                hidden_sizes=prediction_hidden_sizes,
                output_dim=self.latent_dim,
                activation=self._activation,
            )
            for _ in range(self.num_experts)
        )

    def _build_gate(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(input_dim), self.gating_hidden_size),
            self._activation(),
            nn.Linear(self.gating_hidden_size, self.num_experts),
        )

    @staticmethod
    def gate_entropy(gate_probs: torch.Tensor) -> torch.Tensor:
        safe_gate_probs = gate_probs.clamp_min(1e-8)
        return -(safe_gate_probs * safe_gate_probs.log()).sum(dim=-1)

    @staticmethod
    def _mix_outputs(expert_outputs: list[torch.Tensor], gate_probs: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(expert_outputs, dim=1)
        weights = gate_probs.unsqueeze(-1)
        return (stacked * weights).sum(dim=1)

    def _prediction_features_with_info(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_probs = torch.softmax(self.prediction_gate(hidden_state), dim=-1)
        pred_features = self._mix_outputs([expert(hidden_state) for expert in self.prediction_trunk], gate_probs)
        return pred_features, gate_probs

    def _next_hidden_with_info(self, hidden_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.action_embedding(action)
        dynamics_input = torch.cat([hidden_state, action_emb], dim=-1)
        gate_probs = torch.softmax(self.dynamics_gate(dynamics_input), dim=-1)
        next_hidden = self._mix_outputs([expert(dynamics_input) for expert in self.dynamics], gate_probs)
        return self._normalize(next_hidden), gate_probs

    def initial_inference_with_info(self, obs: torch.Tensor) -> tuple[MuZeroInitialOutput, dict[str, torch.Tensor]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.representation(obs_tensor)
        hidden_state = self._normalize(self.representation_head(features))
        pred_features, prediction_gate_probs = self._prediction_features_with_info(hidden_state)
        policy_logits = self.policy_head(pred_features)
        value = self.value_head(pred_features).squeeze(-1)
        return (
            MuZeroInitialOutput(hidden_state=hidden_state, policy_logits=policy_logits, value=value),
            {"prediction_gate_probs": prediction_gate_probs},
        )

    def recurrent_inference_with_info(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[MuZeroRecurrentOutput, dict[str, torch.Tensor]]:
        hidden_tensor = torch.as_tensor(hidden_state, dtype=torch.float32)
        if hidden_tensor.ndim != 2:
            raise ValueError(f"expected hidden_state shape (batch, latent_dim), got {tuple(hidden_tensor.shape)!r}")

        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=hidden_tensor.device)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 1:
            raise ValueError(f"expected action shape (batch,), got {tuple(action_tensor.shape)!r}")
        if int(action_tensor.shape[0]) != int(hidden_tensor.shape[0]):
            raise ValueError(
                f"expected action batch {int(hidden_tensor.shape[0])}, got {int(action_tensor.shape[0])}"
            )

        next_hidden, dynamics_gate_probs = self._next_hidden_with_info(hidden_tensor, action_tensor)
        reward = self.reward_head(next_hidden).squeeze(-1)
        pred_features, prediction_gate_probs = self._prediction_features_with_info(next_hidden)
        policy_logits = self.policy_head(pred_features)
        value = self.value_head(pred_features).squeeze(-1)
        return (
            MuZeroRecurrentOutput(
                hidden_state=next_hidden,
                reward=reward,
                policy_logits=policy_logits,
                value=value,
            ),
            {
                "dynamics_gate_probs": dynamics_gate_probs,
                "prediction_gate_probs": prediction_gate_probs,
            },
        )

    def initial_inference(self, obs: torch.Tensor) -> MuZeroInitialOutput:
        output, _ = self.initial_inference_with_info(obs)
        return output

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> MuZeroRecurrentOutput:
        output, _ = self.recurrent_inference_with_info(hidden_state, action)
        return output
