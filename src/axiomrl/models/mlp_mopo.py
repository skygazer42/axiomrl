from collections.abc import Iterator, Sequence

import torch
from torch import nn

from axiomrl.models.mlp_td3 import _build_mlp

LOGVAR_MIN = -10.0
LOGVAR_MAX = 2.0


class MLPMOPOEnsembleModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        num_ensembles: int = 5,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if int(num_ensembles) < 2:
            raise ValueError(f"num_ensembles must be >= 2, got {num_ensembles}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_ensembles = int(num_ensembles)
        self.output_dim = self.obs_dim + 1

        input_dim = self.obs_dim + self.action_dim
        self.ensembles = nn.ModuleList(
            [
                _build_mlp(
                    input_dim=input_dim,
                    hidden_sizes=hidden_sizes,
                    output_dim=self.output_dim * 2,
                    activation=activation,
                )
                for _ in range(self.num_ensembles)
            ]
        )

    def ensemble_parameters(self) -> Iterator[nn.Parameter]:
        for network in self.ensembles:
            yield from network.parameters()

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

    def predict_distribution(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        action_tensor = self._prepare_actions(actions, device=obs_tensor.device)
        inputs = torch.cat([obs_tensor, action_tensor], dim=-1)

        outputs = [network(inputs) for network in self.ensembles]
        stacked = torch.stack(outputs, dim=0)
        means, raw_logvars = torch.chunk(stacked, 2, dim=-1)
        logvars = raw_logvars.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
        return means, logvars

    def sample_transition(self, obs: object, actions: object) -> dict[str, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        means, logvars = self.predict_distribution(obs_tensor, actions)
        std = torch.exp(0.5 * logvars)

        batch_size = int(obs_tensor.shape[0])
        device = obs_tensor.device
        ensemble_indices = torch.randint(0, self.num_ensembles, (batch_size,), device=device)
        batch_indices = torch.arange(batch_size, device=device)
        chosen_means = means[ensemble_indices, batch_indices]
        chosen_std = std[ensemble_indices, batch_indices]
        samples = chosen_means + torch.randn_like(chosen_std) * chosen_std

        delta_obs = samples[:, : self.obs_dim]
        rewards = samples[:, self.obs_dim]
        disagreement = means.std(dim=0).mean(dim=-1)
        next_obs = obs_tensor + delta_obs

        return {
            "next_obs": next_obs,
            "rewards": rewards,
            "disagreement": disagreement,
        }
