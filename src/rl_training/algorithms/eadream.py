from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.dreamer import Dreamer
from rl_training.models.eadream import EADreamModel


def _event_targets(
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    deltas = (next_obs - obs).abs() / 255.0
    event_intensity = deltas.flatten(start_dim=1).mean(dim=1)
    event_from_pixels = event_intensity >= float(threshold)
    event_from_rewards = rewards.abs() > float(threshold)
    return torch.logical_or(event_from_pixels, event_from_rewards).to(dtype=torch.float32)


class EADream(Dreamer):
    def __init__(
        self,
        *,
        model: EADreamModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        event_loss_coef: float = 0.5,
        event_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        if float(event_loss_coef) < 0.0:
            raise ValueError(f"event_loss_coef must be >= 0, got {event_loss_coef}")
        if float(event_threshold) < 0.0:
            raise ValueError(f"event_threshold must be >= 0, got {event_threshold}")
        self.event_loss_coef = float(event_loss_coef)
        self.event_threshold = float(event_threshold)

    @property
    def eadream_model(self) -> EADreamModel:
        return self.model

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        features = self.eadream_model.encode(obs)
        next_features = self.eadream_model.dynamics_step(features, actions)
        decoded_next = self.eadream_model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.eadream_model.predict_reward(next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        event_targets = _event_targets(obs, next_obs, rewards, threshold=self.event_threshold)
        predicted_event_probs = self.eadream_model.event_probability(next_features)
        event_loss = F.binary_cross_entropy(predicted_event_probs, event_targets)

        loss = reconstruction_loss + reward_loss + self.event_loss_coef * event_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        metrics = {
            "eadream_world_model_loss": float(loss.detach().cpu().item()),
            "eadream_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "eadream_reward_loss": float(reward_loss.detach().cpu().item()),
            "eadream_event_loss": float(event_loss.detach().cpu().item()),
            "eadream_event_prob_mean": float(predicted_event_probs.mean().detach().cpu().item()),
            "eadream_event_rate": float(event_targets.mean().detach().cpu().item()),
            "eadream_event_threshold": self.event_threshold,
            "eadream_event_scale": self.eadream_model.event_scale,
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
            start_features = self.eadream_model.encode(start_obs.to(dtype=torch.float32))

        features = start_features.detach()
        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []
        event_prob_terms: list[torch.Tensor] = []

        for _ in range(horizon):
            logits = self.eadream_model.actor_logits(features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())
            value_terms.append(self.eadream_model.value(features))

            with torch.no_grad():
                next_features = self.eadream_model.dynamics_step(features, actions)
                reward_terms.append(self.eadream_model.predict_reward(next_features))
                event_prob_terms.append(self.eadream_model.event_probability(next_features))
                features = next_features.detach()

        with torch.no_grad():
            bootstrap_value = self.eadream_model.value(features).detach()

        returns: list[torch.Tensor] = []
        running_return = bootstrap_value
        for reward in reversed(reward_terms):
            running_return = reward + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_tensor = torch.stack(returns, dim=0)
        values_tensor = torch.stack(value_terms, dim=0)
        advantages = returns_tensor - values_tensor

        logprobs_tensor = torch.stack(logprob_terms, dim=0)
        entropy_tensor = torch.stack(entropy_terms, dim=0)

        actor_loss = -(logprobs_tensor * advantages.detach()).mean() - self.entropy_coef * entropy_tensor.mean()
        critic_loss = 0.5 * F.mse_loss(values_tensor, returns_tensor.detach())

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        metrics = {
            "eadream_actor_loss": float(actor_loss.detach().cpu().item()),
            "eadream_critic_loss": float(critic_loss.detach().cpu().item()),
            "eadream_imagination_horizon": float(horizon),
            "eadream_imagination_reward_mean": float(torch.stack(reward_terms, dim=0).mean().detach().cpu().item()) if reward_terms else 0.0,
            "eadream_imagined_event_prob_mean": float(torch.stack(event_prob_terms, dim=0).mean().detach().cpu().item()) if event_prob_terms else 0.0,
            "eadream_event_scale": self.eadream_model.event_scale,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["event_loss_coef"] = self.event_loss_coef
        state["event_threshold"] = self.event_threshold
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.event_loss_coef = float(state_dict.get("event_loss_coef", self.event_loss_coef))
        self.event_threshold = float(state_dict.get("event_threshold", self.event_threshold))
