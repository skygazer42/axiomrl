from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rl_training.models.cnn.nature import NatureCNN
from rl_training.models.cnn.q_network import _build_head


class CNNSPRQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
        transition_hidden_size: int = 512,
        projection_dim: int = 256,
        action_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = int(action_dim)
        self.features_dim = int(features_dim)
        self.projection_dim = int(projection_dim)

        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=self.features_dim)
        self.q_head = _build_head(
            input_dim=self.features_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim,
            activation=activation,
        )
        self.action_embedding = nn.Embedding(self.action_dim, int(action_embed_dim))
        self.transition_model = nn.Sequential(
            nn.Linear(self.features_dim + int(action_embed_dim), int(transition_hidden_size)),
            activation(),
            nn.Linear(int(transition_hidden_size), self.features_dim),
            activation(),
        )
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, int(transition_hidden_size)),
            activation(),
            nn.Linear(int(transition_hidden_size), self.projection_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim),
            activation(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def encode(self, obs: object) -> torch.Tensor:
        return self.feature_extractor(self._prepare_obs(obs))

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.int64, device=latent_tensor.device).reshape(-1)
        action_emb = self.action_embedding(action_tensor)
        return self.transition_model(torch.cat([latent_tensor, action_emb], dim=-1))

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projector(torch.as_tensor(latent, dtype=torch.float32))

    def predict_projection(self, latent: torch.Tensor) -> torch.Tensor:
        projected = self.project(latent)
        return self.predictor(projected)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encode(obs)
        return self.q_head(features)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
