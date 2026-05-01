from collections.abc import Iterator, Sequence

import torch
from torch import nn

from axiomrl.models.mlp_td3 import _build_mlp
from axiomrl.policies.base import PolicyOutput

LATENT_LOG_STD_MIN = -4.0
LATENT_LOG_STD_MAX = 4.0
LATENT_CLIP = 0.5


class MLPBCQModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        perturbation_scale: float = 0.05,
        num_action_samples: int = 10,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if int(obs_dim) < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        if int(action_dim) < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if int(num_action_samples) < 1:
            raise ValueError(f"num_action_samples must be >= 1, got {num_action_samples}")
        if float(perturbation_scale) <= 0.0:
            raise ValueError(f"perturbation_scale must be > 0, got {perturbation_scale}")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.perturbation_scale = float(perturbation_scale)
        self.default_num_action_samples = int(num_action_samples)

        encoder_input_dim = self.obs_dim + self.action_dim
        self.encoder_backbone = _build_mlp(
            input_dim=encoder_input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=hidden_sizes[-1],
            activation=activation,
        )
        self.latent_mean = nn.Linear(hidden_sizes[-1], self.latent_dim)
        self.latent_log_std = nn.Linear(hidden_sizes[-1], self.latent_dim)

        self.decoder_net = _build_mlp(
            input_dim=self.obs_dim + self.latent_dim,
            hidden_sizes=hidden_sizes,
            output_dim=self.action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )
        self.perturbation_net = _build_mlp(
            input_dim=encoder_input_dim,
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

    def vae_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.encoder_backbone.parameters()
        yield from self.latent_mean.parameters()
        yield from self.latent_log_std.parameters()
        yield from self.decoder_net.parameters()

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.perturbation_net.parameters()

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

    def _repeat_observations(self, obs_tensor: torch.Tensor, num_action_samples: int) -> torch.Tensor:
        return (
            obs_tensor.unsqueeze(1)
            .expand(-1, num_action_samples, -1)
            .reshape(
                obs_tensor.shape[0] * num_action_samples,
                obs_tensor.shape[1],
            )
        )

    def encode(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        features = self.encoder_backbone(torch.cat([obs_tensor, action_tensor], dim=-1))
        mean = self.latent_mean(features)
        log_std = self.latent_log_std(features).clamp(LATENT_LOG_STD_MIN, LATENT_LOG_STD_MAX)
        return mean, log_std

    def decode(
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
        return self.decoder_net(torch.cat([obs_tensor, latent_tensor], dim=-1))

    def reconstruct(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.encode(obs, actions)
        std = log_std.exp()
        latent = mean + std * torch.randn_like(std)
        reconstructed_actions = self.decode(obs, latent=latent)
        return reconstructed_actions, mean, log_std

    def perturb_actions(self, obs: object, actions: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        deltas = self.perturbation_net(torch.cat([obs_tensor, action_tensor], dim=-1))
        return (action_tensor + self.perturbation_scale * deltas).clamp(-1.0, 1.0)

    def q_values(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def sample_candidate_actions(
        self,
        obs: object,
        *,
        num_action_samples: int | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        sample_count = int(num_action_samples or self.default_num_action_samples)
        repeated_obs = self._repeat_observations(obs_tensor, sample_count)
        decoded_actions = self.decode(repeated_obs, deterministic=deterministic)
        perturbed_actions = self.perturb_actions(repeated_obs, decoded_actions)
        return perturbed_actions.reshape(obs_tensor.shape[0], sample_count, self.action_dim)

    def select_actions(
        self,
        obs: object,
        *,
        num_action_samples: int | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        candidate_actions = self.sample_candidate_actions(
            obs_tensor,
            num_action_samples=num_action_samples,
            deterministic=deterministic,
        )
        repeated_obs = self._repeat_observations(obs_tensor, candidate_actions.shape[1])
        flat_actions = candidate_actions.reshape(-1, self.action_dim)
        q1_values, q2_values = self.q_values(repeated_obs, flat_actions)
        candidate_values = torch.minimum(q1_values, q2_values).reshape(obs_tensor.shape[0], candidate_actions.shape[1])
        best_indices = candidate_values.argmax(dim=1)
        batch_indices = torch.arange(obs_tensor.shape[0], device=obs_tensor.device)
        return candidate_actions[batch_indices, best_indices]

    def actor(
        self,
        obs: object,
        *,
        num_action_samples: int | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        return self.select_actions(
            obs,
            num_action_samples=num_action_samples,
            deterministic=deterministic,
        )

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = True,
    ) -> PolicyOutput:
        del state
        return PolicyOutput(
            actions=self.actor(obs, deterministic=deterministic),
            logprobs=None,
            values=None,
            entropy=None,
            state=None,
        )
