from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.dreamer import Dreamer
from axiomrl.models.dreamer import DreamerModel


class Twisted(Dreamer):
    def __init__(
        self,
        *,
        model: DreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        reuse_loss_coef: float = 0.5,
        reuse_threshold: float = 0.03,
        transport_temperature: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        if float(reuse_loss_coef) < 0.0:
            raise ValueError(f"reuse_loss_coef must be >= 0, got {reuse_loss_coef}")
        if float(reuse_threshold) < 0.0:
            raise ValueError(f"reuse_threshold must be >= 0, got {reuse_threshold}")
        if float(transport_temperature) <= 0.0:
            raise ValueError(f"transport_temperature must be > 0, got {transport_temperature}")
        self.reuse_loss_coef = float(reuse_loss_coef)
        self.reuse_threshold = float(reuse_threshold)
        self.transport_temperature = float(transport_temperature)

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_base = obs / 255.0
        obs_target = next_obs / 255.0
        features = self.model.encode(obs)
        next_features = self.model.dynamics_step(features, actions)
        decoded_next = self.model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.model.predict_reward(next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        frame_delta = (obs_target - obs_base).abs()
        reuse_weights = torch.sigmoid((self.reuse_threshold - frame_delta) / self.transport_temperature)
        reuse_residual = (decoded_next - obs_base).pow(2) * reuse_weights
        reuse_loss = reuse_residual.sum() / reuse_weights.sum().clamp_min(1.0)

        loss = reconstruction_loss + reward_loss + self.reuse_loss_coef * reuse_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        metrics = {
            "twisted_world_model_loss": float(loss.detach().cpu().item()),
            "twisted_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "twisted_reward_loss": float(reward_loss.detach().cpu().item()),
            "twisted_reuse_loss": float(reuse_loss.detach().cpu().item()),
            "twisted_reuse_loss_coef": self.reuse_loss_coef,
            "twisted_reuse_threshold": self.reuse_threshold,
            "twisted_transport_temperature": self.transport_temperature,
            "twisted_transport_weight_mean": float(reuse_weights.mean().detach().cpu().item()),
            "twisted_persistence_rate": float(
                (frame_delta <= self.reuse_threshold).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def update_actor_critic(
        self,
        start_obs: torch.Tensor,
        *,
        imagination_horizon: int,
        global_step: int,
    ) -> UpdateResult:
        result = super().update_actor_critic(
            start_obs,
            imagination_horizon=imagination_horizon,
            global_step=global_step,
        )
        metrics = {key.replace("dreamer_", "twisted_"): value for key, value in result.metrics.items()}
        metrics["twisted_reuse_loss_coef"] = self.reuse_loss_coef
        return UpdateResult(metrics=metrics, num_gradient_steps=result.num_gradient_steps)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["reuse_loss_coef"] = self.reuse_loss_coef
        state["reuse_threshold"] = self.reuse_threshold
        state["transport_temperature"] = self.transport_temperature
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.reuse_loss_coef = float(state_dict.get("reuse_loss_coef", self.reuse_loss_coef))
        self.reuse_threshold = float(state_dict.get("reuse_threshold", self.reuse_threshold))
        self.transport_temperature = float(state_dict.get("transport_temperature", self.transport_temperature))
