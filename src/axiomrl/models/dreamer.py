from collections.abc import Iterator, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.policies.base import PolicyOutput


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


class DreamerModel(nn.Module):
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
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"DreamerModel expects channel-first image observations, got {self.obs_shape!r}")

        channels, height, width = self.obs_shape
        if channels < 1:
            raise ValueError(f"obs_shape must have >= 1 channels, got {self.obs_shape!r}")
        if height < 36 or width < 36:
            raise ValueError(f"obs_shape must be at least 36x36 for NatureCNN, got {self.obs_shape!r}")

        self.action_dim = int(action_dim)
        self.features_dim = int(features_dim)
        self.action_embed_dim = int(action_embed_dim)

        self.encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=self.features_dim)
        self.action_embedding = nn.Embedding(self.action_dim, self.action_embed_dim)
        self.dynamics = nn.GRUCell(self.action_embed_dim, self.features_dim)

        conv_h, conv_w = _nature_conv_output_hw(height, width)
        if conv_h < 1 or conv_w < 1:
            raise ValueError(f"invalid convolution output for obs_shape={self.obs_shape!r}")

        self._decoder_h = int(conv_h)
        self._decoder_w = int(conv_w)
        self.decoder_fc = nn.Linear(self.features_dim, 64 * self._decoder_h * self._decoder_w)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=8, stride=4),
        )

        self.reward_head = self._build_mlp(self.features_dim, reward_hidden_sizes, 1)
        self.actor_head = self._build_mlp(self.features_dim, actor_hidden_sizes, self.action_dim)
        self.critic_head = self._build_mlp(self.features_dim, critic_hidden_sizes, 1)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        return nn.Sequential(*layers)

    def _prepare_obs(self, obs: object, *, device: torch.device) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def dynamics_step(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=features.device)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        action_embed = self.action_embedding(action_tensor)
        return self.dynamics(action_embed, features)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = int(features.shape[0])
        hidden = self.decoder_fc(features).view(batch_size, 64, self._decoder_h, self._decoder_w)
        decoded = self.decoder(hidden)
        return torch.sigmoid(decoded)

    def predict_reward(self, features: torch.Tensor) -> torch.Tensor:
        return self.reward_head(features).squeeze(-1)

    def actor_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_head(features)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_head(features).squeeze(-1)

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
        logits = self.actor_logits(features)
        distribution = Categorical(logits=logits)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value(features)
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

    def parameters_actor(self) -> Iterator[nn.Parameter]:
        yield from self.actor_head.parameters()

    def parameters_critic(self) -> Iterator[nn.Parameter]:
        yield from self.critic_head.parameters()
