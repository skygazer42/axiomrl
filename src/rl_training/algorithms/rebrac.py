from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_td3 import MLPTD3Model


def _rebrac_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    actor_q_values = torch.as_tensor(batch["actor_q_values"], dtype=torch.float32, device=q1_values.device)
    actor_bc_loss = torch.as_tensor(batch["actor_bc_loss"], dtype=torch.float32, device=q1_values.device)
    critic_bc_penalty = torch.as_tensor(batch["critic_bc_penalty"], dtype=torch.float32, device=q1_values.device)
    actor_bc_weight = torch.as_tensor(batch["actor_bc_weight"], dtype=torch.float32, device=q1_values.device)
    critic_bc_weight = torch.as_tensor(batch["critic_bc_weight"], dtype=torch.float32, device=q1_values.device)
    actor_q_weight = torch.as_tensor(batch["actor_q_weight"], dtype=torch.float32, device=q1_values.device)

    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    actor_loss = actor_bc_weight * actor_bc_loss - actor_q_weight * actor_q_values.mean()
    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "actor_bc_loss": actor_bc_loss,
        "actor_q_mean": actor_q_values.mean(),
        "critic_bc_penalty": critic_bc_penalty,
        "actor_bc_weight": actor_bc_weight,
        "critic_bc_weight": critic_bc_weight,
        "actor_q_weight": actor_q_weight,
    }


def rebrac_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _rebrac_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class ReBRAC:
    def __init__(
        self,
        *,
        model: MLPTD3Model,
        learning_rate: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        actor_bc_weight: float,
        critic_bc_weight: float,
        actor_q_weight: float,
    ) -> None:
        if int(policy_delay) < 1:
            raise ValueError(f"policy_delay must be >= 1, got {policy_delay}")
        if float(actor_bc_weight) <= 0.0:
            raise ValueError(f"actor_bc_weight must be > 0, got {actor_bc_weight}")
        if float(critic_bc_weight) < 0.0:
            raise ValueError(f"critic_bc_weight must be >= 0, got {critic_bc_weight}")
        if float(actor_q_weight) <= 0.0:
            raise ValueError(f"actor_q_weight must be > 0, got {actor_q_weight}")

        # This narrow package version stays close to TD3+BC while exposing the
        # fixed behavior-regularization knobs that make ReBRAC a distinct
        # offline baseline on the same runtime surface.
        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=float(learning_rate))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.actor_bc_weight = float(actor_bc_weight)
        self.critic_bc_weight = float(critic_bc_weight)
        self.actor_q_weight = float(actor_q_weight)
        self.update_count = 0

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
        # Fall back to current behavior actions when explicit next-state
        # actions are unavailable. Random datasets and Minari-backed payloads
        # now provide this field directly.
        next_actions = torch.as_tensor(batch.get("next_actions", batch["actions"]), dtype=torch.float32)

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_policy_actions = self.target_model.actor(next_obs)
            noise = torch.randn_like(next_policy_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_policy_actions = (next_policy_actions + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_policy_actions)
            critic_bc_penalty = F.mse_loss(next_policy_actions, next_actions, reduction="none").mean(dim=-1)
            target_q_values = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(target_q1, target_q2) - self.critic_bc_weight * critic_bc_penalty
            )

        actor_bc_weight_tensor = torch.as_tensor(self.actor_bc_weight, dtype=torch.float32, device=obs.device)
        critic_bc_weight_tensor = torch.as_tensor(self.critic_bc_weight, dtype=torch.float32, device=obs.device)
        actor_q_weight_tensor = torch.as_tensor(self.actor_q_weight, dtype=torch.float32, device=obs.device)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _rebrac_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "actor_q_values": current_q1.detach(),
                "actor_bc_loss": torch.zeros((), dtype=torch.float32, device=obs.device),
                "critic_bc_penalty": critic_bc_penalty.mean(),
                "actor_bc_weight": actor_bc_weight_tensor,
                "critic_bc_weight": critic_bc_weight_tensor,
                "actor_q_weight": actor_q_weight_tensor,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        actor_loss_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        actor_bc_loss_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        actor_q_mean_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        self.update_count += 1

        if self.update_count % self.policy_delay == 0:
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_actions = self.model.actor(obs)
            actor_q1, _ = self.model.q_values(obs, policy_actions)
            actor_bc_loss = F.mse_loss(policy_actions, actions)
            actor_terms = _rebrac_loss_terms(
                {
                    "q1_values": current_q1.detach(),
                    "q2_values": current_q2.detach(),
                    "target_q_values": target_q_values.detach(),
                    "actor_q_values": actor_q1,
                    "actor_bc_loss": actor_bc_loss,
                    "critic_bc_penalty": critic_bc_penalty.mean().detach(),
                    "actor_bc_weight": actor_bc_weight_tensor,
                    "critic_bc_weight": critic_bc_weight_tensor,
                    "actor_q_weight": actor_q_weight_tensor,
                }
            )
            actor_terms["actor_loss"].backward()
            self.actor_optimizer.step()
            self.soft_update_targets()
            actor_loss_value = actor_terms["actor_loss"].detach()
            actor_bc_loss_value = actor_terms["actor_bc_loss"].detach()
            actor_q_mean_value = actor_terms["actor_q_mean"].detach()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_loss_value.cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "actor_bc_loss": float(actor_bc_loss_value.cpu().item()),
            "actor_q_mean": float(actor_q_mean_value.cpu().item()),
            "critic_bc_penalty": float(critic_terms["critic_bc_penalty"].detach().cpu().item()),
            "actor_bc_weight": float(actor_bc_weight_tensor.detach().cpu().item()),
            "critic_bc_weight": float(critic_bc_weight_tensor.detach().cpu().item()),
            "actor_q_weight": float(actor_q_weight_tensor.detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.update_count = int(state_dict.get("update_count", 0))

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
