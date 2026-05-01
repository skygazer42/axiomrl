from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.mlp_noisy_q_network import NoisyLinear


def _build_noisy_head(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
    sigma_init: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.append(NoisyLinear(last_dim, int(hidden_dim), sigma_init=sigma_init))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(NoisyLinear(last_dim, int(output_dim), sigma_init=sigma_init))
    return nn.Sequential(*layers)


class CNNDuelingNoisyQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        sigma_init: float = 0.5,
        features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"CNNDuelingNoisyQNetwork expects 3D channel-first observations, got {self.obs_shape!r}")

        self.action_dim = int(action_dim)
        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=int(features_dim))

        if hidden_sizes:
            trunk_hidden_sizes = tuple(int(dim) for dim in hidden_sizes[:-1])
            head_input_dim = int(hidden_sizes[-1])
            self.trunk = _build_noisy_head(
                input_dim=int(features_dim),
                hidden_sizes=trunk_hidden_sizes,
                output_dim=head_input_dim,
                activation=activation,
                sigma_init=sigma_init,
            )
            self.trunk_activation: nn.Module = activation()
        else:
            head_input_dim = int(features_dim)
            self.trunk = nn.Identity()
            self.trunk_activation = nn.Identity()

        self.value_head = NoisyLinear(head_input_dim, 1, sigma_init=sigma_init)
        self.advantage_head = NoisyLinear(head_input_dim, self.action_dim, sigma_init=sigma_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.feature_extractor(obs_tensor)
        trunk_features = self.trunk_activation(self.trunk(features))
        values = self.value_head(trunk_features)
        advantages = self.advantage_head(trunk_features)
        centered_advantages = advantages - advantages.mean(dim=-1, keepdim=True)
        return values + centered_advantages

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
