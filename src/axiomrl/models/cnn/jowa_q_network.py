from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.cnn.q_network import _build_head


def _conv_output_size(size: int, *, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


def _nature_conv_output_hw(height: int, width: int) -> tuple[int, int]:
    h = _conv_output_size(height, kernel=8, stride=4)
    w = _conv_output_size(width, kernel=8, stride=4)
    h = _conv_output_size(h, kernel=4, stride=2)
    w = _conv_output_size(w, kernel=4, stride=2)
    h = _conv_output_size(h, kernel=3, stride=1)
    w = _conv_output_size(w, kernel=3, stride=1)
    return h, w


class CNNJOWAQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
        transition_hidden_size: int = 512,
        reward_hidden_size: int = 256,
        action_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = int(action_dim)
        self.features_dim = int(features_dim)

        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNJOWAQNetwork expects channel-first image observations, got {self.obs_shape!r}")

        channels, height, width = self.obs_shape
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
        self.reward_head = nn.Sequential(
            nn.Linear(self.features_dim, int(reward_hidden_size)),
            activation(),
            nn.Linear(int(reward_hidden_size), 1),
        )

        decoder_h, decoder_w = _nature_conv_output_hw(height, width)
        self._decoder_h = int(decoder_h)
        self._decoder_w = int(decoder_w)
        self.decoder_fc = nn.Linear(self.features_dim, 64 * self._decoder_h * self._decoder_w)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            activation(),
            nn.ConvTranspose2d(32, channels, kernel_size=8, stride=4),
            nn.Sigmoid(),
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

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        return self.reward_head(latent_tensor).squeeze(-1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32)
        batch_size = int(latent_tensor.shape[0])
        hidden = self.decoder_fc(latent_tensor).view(batch_size, 64, self._decoder_h, self._decoder_w)
        return self.decoder(hidden)

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
