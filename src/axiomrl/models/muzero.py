from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN


def _build_mlp(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class MuZeroInitialOutput:
    hidden_state: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class MuZeroRecurrentOutput:
    hidden_state: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor


class MuZeroModel(nn.Module):
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
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"MuZeroModel expects 3D channel-first observations, got {self.obs_shape!r}")

        self.action_dim = int(action_dim)
        if self.action_dim < 2:
            raise ValueError(f"action_dim must be >= 2, got {self.action_dim}")

        self.latent_dim = int(latent_dim)
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}")

        self.action_embed_dim = int(action_embed_dim)
        if self.action_embed_dim < 1:
            raise ValueError(f"action_embed_dim must be >= 1, got {self.action_embed_dim}")

        self.normalize_latent = bool(normalize_latent)

        features_dim = int(representation_features_dim or self.latent_dim)
        self.representation = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        if features_dim != self.latent_dim:
            self.representation_head = nn.Linear(features_dim, self.latent_dim)
        else:
            self.representation_head = nn.Identity()

        self.action_embedding = nn.Embedding(self.action_dim, self.action_embed_dim)
        self.dynamics = _build_mlp(
            input_dim=self.latent_dim + self.action_embed_dim,
            hidden_sizes=dynamics_hidden_sizes,
            output_dim=self.latent_dim,
            activation=activation,
        )
        self.reward_head = _build_mlp(
            input_dim=self.latent_dim,
            hidden_sizes=prediction_hidden_sizes,
            output_dim=1,
            activation=activation,
        )

        self.prediction_trunk = _build_mlp(
            input_dim=self.latent_dim,
            hidden_sizes=prediction_hidden_sizes,
            output_dim=self.latent_dim,
            activation=activation,
        )
        self.policy_head = nn.Linear(self.latent_dim, self.action_dim)
        self.value_head = nn.Linear(self.latent_dim, 1)

    def _normalize(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latent:
            return latent
        scale = latent.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        return latent / scale

    def initial_inference(self, obs: torch.Tensor) -> MuZeroInitialOutput:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.representation(obs_tensor)
        hidden_state = self._normalize(self.representation_head(features))
        pred_features = self.prediction_trunk(hidden_state)
        policy_logits = self.policy_head(pred_features)
        value = self.value_head(pred_features).squeeze(-1)
        return MuZeroInitialOutput(hidden_state=hidden_state, policy_logits=policy_logits, value=value)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> MuZeroRecurrentOutput:
        hidden_tensor = torch.as_tensor(hidden_state, dtype=torch.float32)
        if hidden_tensor.ndim != 2:
            raise ValueError(f"expected hidden_state shape (batch, latent_dim), got {tuple(hidden_tensor.shape)!r}")

        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=hidden_tensor.device)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 1:
            raise ValueError(f"expected action shape (batch,), got {tuple(action_tensor.shape)!r}")
        if int(action_tensor.shape[0]) != int(hidden_tensor.shape[0]):
            raise ValueError(f"expected action batch {int(hidden_tensor.shape[0])}, got {int(action_tensor.shape[0])}")

        action_emb = self.action_embedding(action_tensor)
        dynamics_input = torch.cat([hidden_tensor, action_emb], dim=-1)
        next_hidden = self._normalize(self.dynamics(dynamics_input))

        reward = self.reward_head(next_hidden).squeeze(-1)
        pred_features = self.prediction_trunk(next_hidden)
        policy_logits = self.policy_head(pred_features)
        value = self.value_head(pred_features).squeeze(-1)
        return MuZeroRecurrentOutput(
            hidden_state=next_hidden,
            reward=reward,
            policy_logits=policy_logits,
            value=value,
        )
