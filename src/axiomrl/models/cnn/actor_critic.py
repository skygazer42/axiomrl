from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.policies.base import PolicyOutput


def _build_head(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class CNNActorCritic(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        hidden_sizes: Sequence[int] = (512,),
        activation: type[nn.Module] = nn.ReLU,
        features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.feature_extractor = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        self.actor = _build_head(
            input_dim=features_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )
        self.critic = _build_head(
            input_dim=features_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _distribution(self, obs: object) -> tuple[Categorical, torch.Tensor]:
        obs_tensor = self._prepare_obs(obs)
        features = self.feature_extractor(obs_tensor)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return Categorical(logits=logits), values

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        del state
        distribution, values = self._distribution(obs)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return PolicyOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            entropy=entropy,
            state=None,
        )

    def evaluate_actions(self, obs: object, actions: object) -> dict[str, torch.Tensor]:
        distribution, values = self._distribution(obs)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=values.device)
        return {
            "logprobs": distribution.log_prob(action_tensor),
            "entropy": distribution.entropy(),
            "values": values,
        }
