from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.diamond import Diamond
from axiomrl.models.dreamer import DreamerModel


def _build_horizon_schedule(
    horizon: int,
    *,
    schedule_bias: float,
    subframe_budget_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if horizon <= 1:
        return torch.ones(1, device=device, dtype=torch.float32)

    positions = torch.linspace(0.0, 1.0, steps=horizon, device=device, dtype=torch.float32)
    centered = positions - positions.mean()
    budget_scale = max(float(subframe_budget_ratio), 1e-3)
    logits = float(schedule_bias) * centered / budget_scale
    return torch.softmax(logits, dim=0)


class HorizonImagination(Diamond):
    def __init__(
        self,
        *,
        model: DreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        denoising_loss_coef: float = 0.5,
        noise_scale: float = 0.15,
        denoiser_hidden_channels: int = 64,
        stabilization_coef: float = 0.25,
        schedule_bias: float = 0.5,
        subframe_budget_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
            denoising_loss_coef=denoising_loss_coef,
            noise_scale=noise_scale,
            denoiser_hidden_channels=denoiser_hidden_channels,
        )
        if float(stabilization_coef) < 0.0:
            raise ValueError(f"stabilization_coef must be >= 0, got {stabilization_coef}")
        if not math.isfinite(float(schedule_bias)):
            raise ValueError(f"schedule_bias must be finite, got {schedule_bias}")
        if not 0.0 <= float(subframe_budget_ratio) <= 1.0:
            raise ValueError(f"subframe_budget_ratio must be in [0, 1], got {subframe_budget_ratio}")

        self.stabilization_coef = float(stabilization_coef)
        self.schedule_bias = float(schedule_bias)
        self.subframe_budget_ratio = float(subframe_budget_ratio)

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        features = self.model.encode(obs)
        next_features = self.model.dynamics_step(features, actions)
        decoded_next = self.model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.model.predict_reward(next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        if self.noise_scale > 0.0:
            noise = torch.randn_like(obs_target) * self.noise_scale
            noisy_target = (obs_target + noise).clamp(0.0, 1.0)
        else:
            noisy_target = obs_target
        denoised_next = self.denoiser(decoded_next, noisy_target)
        denoising_loss = F.mse_loss(denoised_next, obs_target)

        with torch.no_grad():
            encoded_next_features = self.model.encode(next_obs)
        predicted_delta = F.normalize(next_features - features.detach(), dim=-1)
        target_delta = F.normalize(encoded_next_features - features.detach(), dim=-1)
        stabilization_loss = F.smooth_l1_loss(predicted_delta, target_delta)

        loss = (
            reconstruction_loss
            + reward_loss
            + self.denoising_loss_coef * denoising_loss
            + self.stabilization_coef * stabilization_loss
        )
        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.denoiser_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()
        self.denoiser_optimizer.step()

        metrics = {
            "horizon_imagination_world_model_loss": float(loss.detach().cpu().item()),
            "horizon_imagination_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "horizon_imagination_reward_loss": float(reward_loss.detach().cpu().item()),
            "horizon_imagination_denoising_loss": float(denoising_loss.detach().cpu().item()),
            "horizon_imagination_stabilization_loss": float(stabilization_loss.detach().cpu().item()),
            "horizon_imagination_denoising_loss_coef": self.denoising_loss_coef,
            "horizon_imagination_stabilization_coef": self.stabilization_coef,
            "horizon_imagination_noise_scale": self.noise_scale,
            "horizon_imagination_subframe_budget_ratio": self.subframe_budget_ratio,
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
        device = start_obs.device
        schedule = _build_horizon_schedule(
            horizon,
            schedule_bias=self.schedule_bias,
            subframe_budget_ratio=self.subframe_budget_ratio,
            device=device,
        )

        with torch.no_grad():
            start_features = self.model.encode(start_obs.to(dtype=torch.float32))

        features = start_features.detach()
        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []

        for _ in range(horizon):
            logits = self.model.actor_logits(features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())
            value_terms.append(self.model.value(features))

            with torch.no_grad():
                next_features = self.model.dynamics_step(features, actions)
                reward_terms.append(self.model.predict_reward(next_features))
                features = next_features.detach()

        with torch.no_grad():
            bootstrap_value = self.model.value(features).detach()

        returns: list[torch.Tensor] = []
        running_return = bootstrap_value
        for reward in reversed(reward_terms):
            running_return = reward + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_tensor = torch.stack(returns, dim=0)
        values_tensor = torch.stack(value_terms, dim=0)
        logprobs_tensor = torch.stack(logprob_terms, dim=0)
        entropy_tensor = torch.stack(entropy_terms, dim=0)
        advantages = returns_tensor - values_tensor
        step_weights = schedule.unsqueeze(-1)

        actor_loss = (
            -((step_weights * logprobs_tensor * advantages.detach()).sum(dim=0).mean())
            - self.entropy_coef * (step_weights * entropy_tensor).sum(dim=0).mean()
        )
        critic_loss = 0.5 * (step_weights * (values_tensor - returns_tensor.detach()).pow(2)).sum(dim=0).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        horizon_positions = torch.linspace(
            1.0 / horizon,
            1.0,
            steps=horizon,
            device=device,
            dtype=torch.float32,
        )
        schedule_mean = float((schedule * horizon_positions).sum().detach().cpu().item())
        schedule_entropy = float((-(schedule * schedule.clamp_min(1e-8).log()).sum()).detach().cpu().item())

        metrics = {
            "horizon_imagination_actor_loss": float(actor_loss.detach().cpu().item()),
            "horizon_imagination_critic_loss": float(critic_loss.detach().cpu().item()),
            "horizon_imagination_imagination_horizon": float(horizon),
            "horizon_imagination_imagination_reward_mean": float(
                torch.stack(reward_terms, dim=0).mean().detach().cpu().item()
            )
            if reward_terms
            else 0.0,
            "horizon_imagination_schedule_mean": schedule_mean,
            "horizon_imagination_schedule_peak": float(schedule.max().detach().cpu().item()),
            "horizon_imagination_schedule_entropy": schedule_entropy,
            "horizon_imagination_schedule_bias": self.schedule_bias,
            "horizon_imagination_subframe_budget_ratio": self.subframe_budget_ratio,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["stabilization_coef"] = self.stabilization_coef
        state["schedule_bias"] = self.schedule_bias
        state["subframe_budget_ratio"] = self.subframe_budget_ratio
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.stabilization_coef = float(state_dict.get("stabilization_coef", self.stabilization_coef))
        self.schedule_bias = float(state_dict.get("schedule_bias", self.schedule_bias))
        self.subframe_budget_ratio = float(state_dict.get("subframe_budget_ratio", self.subframe_budget_ratio))
