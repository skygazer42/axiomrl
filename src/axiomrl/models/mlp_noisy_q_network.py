from collections.abc import Sequence

import torch
from torch import nn


def _scale_noise(epsilon: torch.Tensor) -> torch.Tensor:
    return epsilon.sign() * epsilon.abs().sqrt()


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        mu_range = 1.0 / (self.in_features**0.5)
        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))

        self.register_buffer("epsilon_input", torch.zeros(self.in_features))
        self.register_buffer("epsilon_output", torch.zeros(self.out_features))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma = float(sigma_init) * mu_range
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self) -> None:
        device = self.epsilon_input.device
        self.epsilon_input.copy_(torch.randn(self.in_features, device=device))
        self.epsilon_output.copy_(torch.randn(self.out_features, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.reset_noise()
            epsilon_in = _scale_noise(self.epsilon_input)
            epsilon_out = _scale_noise(self.epsilon_output)
            weight_epsilon = epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0)
            bias_epsilon = epsilon_out
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
            return nn.functional.linear(inputs, weight, bias)

        return nn.functional.linear(inputs, self.weight_mu, self.bias_mu)


class MLPNoisyQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)

        layers: list[nn.Module] = []
        last_dim = int(obs_dim)
        for hidden_dim in hidden_sizes:
            layers.append(NoisyLinear(last_dim, int(hidden_dim), sigma_init=sigma_init))
            layers.append(activation())
            last_dim = int(hidden_dim)
        layers.append(NoisyLinear(last_dim, self.action_dim, sigma_init=sigma_init))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return self.network(obs_tensor)

    def act(self, obs: object, *, epsilon: float = 0.0) -> torch.Tensor:
        q_values = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        greedy_actions = q_values.argmax(dim=-1)

        if epsilon <= 0.0:
            return greedy_actions

        random_actions = torch.randint(0, self.action_dim, greedy_actions.shape, device=greedy_actions.device)
        explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < epsilon
        return torch.where(explore_mask, random_actions, greedy_actions)
