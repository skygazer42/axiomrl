from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from axiomrl.policies.base import PolicyOutput


def _build_backbone(
    *,
    input_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    return nn.Sequential(*layers), last_dim


class MLPPPGModel(nn.Module):
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

        self.backbone, features_dim = _build_backbone(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.policy_head = nn.Linear(features_dim, action_dim)
        self.value_head = nn.Linear(features_dim, 1)
        self.auxiliary_value_head = nn.Linear(features_dim, 1)

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _features(self, obs: object) -> torch.Tensor:
        obs_tensor = self._prepare_obs(obs)
        return self.backbone(obs_tensor)

    def policy_logits(self, obs: object) -> torch.Tensor:
        return self.policy_head(self._features(obs))

    def values(self, obs: object) -> torch.Tensor:
        return self.value_head(self._features(obs)).squeeze(-1)

    def auxiliary_values(self, obs: object) -> torch.Tensor:
        return self.auxiliary_value_head(self._features(obs)).squeeze(-1)

    def _distribution(self, obs: object) -> tuple[Categorical, torch.Tensor]:
        features = self._features(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
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
            "logits": distribution.logits,
        }
