from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from axiomrl.models.mlp_td3 import _build_mlp


class MLPARSModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_net = _build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
            final_activation=nn.Tanh(),
        )

    @property
    def num_parameters(self) -> int:
        return int(self.flat_parameters().numel())

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def actor(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.policy_net(obs_tensor)

    def flat_parameters(self) -> torch.Tensor:
        return parameters_to_vector(self.parameters())

    def set_flat_parameters(self, flat_parameters: torch.Tensor) -> None:
        vector = flat_parameters.to(device=self.flat_parameters().device, dtype=self.flat_parameters().dtype)
        vector_to_parameters(vector, self.parameters())
