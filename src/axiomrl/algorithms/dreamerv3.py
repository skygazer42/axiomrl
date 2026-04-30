from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.dreamer import Dreamer
from axiomrl.models.dreamer import DreamerModel
from axiomrl.policies.base import PolicyOutput


def _symlog(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(values.abs())


def _symexp(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.expm1(values.abs())


def _apply_unimix(logits: torch.Tensor, *, ratio: float) -> Categorical:
    probs = torch.softmax(logits, dim=-1)
    if float(ratio) > 0.0:
        uniform = torch.full_like(probs, 1.0 / float(probs.shape[-1]))
        probs = (1.0 - float(ratio)) * probs + float(ratio) * uniform
    return Categorical(probs=probs)


class DreamerV3(Dreamer):
    def __init__(
        self,
        *,
        model: DreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        unimix_ratio: float = 0.01,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        if not 0.0 <= float(unimix_ratio) < 1.0:
            raise ValueError(f"unimix_ratio must be in [0, 1), got {unimix_ratio}")
        self.unimix_ratio = float(unimix_ratio)

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        del state
        device = next(self.model.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == len(self.model.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)

        features = self.model.encode(obs_tensor)
        logits = self.model.actor_logits(features)
        distribution = _apply_unimix(logits, ratio=self.unimix_ratio)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = _symexp(self.model.value(features))
        return PolicyOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            entropy=entropy,
            state=None,
        )

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        reward_target = _symlog(rewards)

        features = self.model.encode(obs)
        next_features = self.model.dynamics_step(features, actions)
        decoded_next = self.model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.model.predict_reward(next_features)
        reward_loss = F.mse_loss(predicted_rewards, reward_target)

        loss = reconstruction_loss + reward_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        metrics = {
            "dreamerv3_world_model_loss": float(loss.detach().cpu().item()),
            "dreamerv3_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "dreamerv3_reward_symlog_loss": float(reward_loss.detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def update_actor_critic(
        self,
        start_obs: torch.Tensor,
        *,
        imagination_horizon: int,
        global_step: int,
    ) -> UpdateResult:
        del global_step
        self.set_train_mode()

        horizon = max(1, int(imagination_horizon))
        with torch.no_grad():
            start_features = self.model.encode(start_obs.to(dtype=torch.float32))

        features = start_features.detach()
        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_symlog_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []

        for _ in range(horizon):
            logits = self.model.actor_logits(features)
            distribution = _apply_unimix(logits, ratio=self.unimix_ratio)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())
            value_symlog_terms.append(self.model.value(features))

            with torch.no_grad():
                next_features = self.model.dynamics_step(features, actions)
                reward_terms.append(_symexp(self.model.predict_reward(next_features)))
                features = next_features.detach()

        with torch.no_grad():
            bootstrap_value = _symexp(self.model.value(features)).detach()

        returns: list[torch.Tensor] = []
        running_return = bootstrap_value
        for reward in reversed(reward_terms):
            running_return = reward + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_tensor = torch.stack(returns, dim=0)
        returns_symlog = _symlog(returns_tensor)
        values_symlog_tensor = torch.stack(value_symlog_terms, dim=0)
        values_tensor = _symexp(values_symlog_tensor)
        advantages = returns_tensor - values_tensor

        logprobs_tensor = torch.stack(logprob_terms, dim=0)
        entropy_tensor = torch.stack(entropy_terms, dim=0)

        actor_loss = -(logprobs_tensor * advantages.detach()).mean() - self.entropy_coef * entropy_tensor.mean()
        critic_loss = 0.5 * F.mse_loss(values_symlog_tensor, returns_symlog.detach())

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        metrics = {
            "dreamerv3_actor_loss": float(actor_loss.detach().cpu().item()),
            "dreamerv3_critic_loss": float(critic_loss.detach().cpu().item()),
            "dreamerv3_imagination_horizon": float(horizon),
            "dreamerv3_imagination_reward_mean": float(torch.stack(reward_terms, dim=0).mean().detach().cpu().item())
            if reward_terms
            else 0.0,
            "dreamerv3_unimix_ratio": self.unimix_ratio,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["unimix_ratio"] = self.unimix_ratio
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.unimix_ratio = float(state_dict.get("unimix_ratio", self.unimix_ratio))
