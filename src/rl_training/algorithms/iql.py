from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_iql import MLPIQLModel


def _iql_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    value_predictions = torch.as_tensor(batch["value_predictions"], dtype=torch.float32, device=q1_values.device)
    target_state_values = torch.as_tensor(batch["target_state_values"], dtype=torch.float32, device=q1_values.device)
    behavior_logprobs = torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32, device=q1_values.device)
    advantage_weights = torch.as_tensor(batch["advantage_weights"], dtype=torch.float32, device=q1_values.device)
    expectile = float(batch.get("expectile", 0.7))

    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    advantage = target_state_values - value_predictions
    value_weights = torch.where(
        advantage > 0,
        torch.full_like(advantage, expectile),
        torch.full_like(advantage, 1.0 - expectile),
    )
    value_loss = (value_weights * advantage.pow(2)).mean()
    actor_loss = -(advantage_weights * behavior_logprobs).mean()

    return {
        "critic_loss": critic_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "advantage_weight_mean": advantage_weights.mean(),
    }


def iql_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _iql_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class IQL:
    def __init__(
        self,
        *,
        model: MLPIQLModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        expectile: float,
        beta: float,
        max_advantage_weight: float,
    ) -> None:
        if not 0.0 < float(expectile) < 1.0:
            raise ValueError(f"expectile must be in (0, 1), got {expectile}")
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if float(max_advantage_weight) <= 0.0:
            raise ValueError(f"max_advantage_weight must be > 0, got {max_advantage_weight}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=float(learning_rate))
        self.value_optimizer = torch.optim.Adam(self.model.value_parameters(), lr=float(learning_rate))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.expectile = float(expectile)
        self.beta = float(beta)
        self.max_advantage_weight = float(max_advantage_weight)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_values = self.model.value(next_obs)
            target_q_values = rewards + self.gamma * (1.0 - dones) * next_values

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _iql_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "value_predictions": torch.zeros_like(rewards),
                "target_state_values": torch.zeros_like(rewards),
                "behavior_logprobs": torch.zeros_like(rewards),
                "advantage_weights": torch.ones_like(rewards),
                "expectile": self.expectile,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            target_q1, target_q2 = self.target_model.q_values(obs, actions)
            target_state_values = torch.minimum(target_q1, target_q2)

        value_predictions = self.model.value(obs)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_terms = _iql_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "value_predictions": value_predictions,
                "target_state_values": target_state_values,
                "behavior_logprobs": torch.zeros_like(rewards),
                "advantage_weights": torch.ones_like(rewards),
                "expectile": self.expectile,
            }
        )
        value_terms["value_loss"].backward()
        self.value_optimizer.step()

        with torch.no_grad():
            advantages = target_state_values - value_predictions.detach()
            advantage_weights = torch.exp(advantages / self.beta).clamp(max=self.max_advantage_weight)

        behavior_logprobs = self.model.action_logprobs(obs, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _iql_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "value_predictions": value_predictions.detach(),
                "target_state_values": target_state_values.detach(),
                "behavior_logprobs": behavior_logprobs,
                "advantage_weights": advantage_weights,
                "expectile": self.expectile,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "value_loss": float(value_terms["value_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "advantage_weight_mean": float(actor_terms["advantage_weight_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.q1.parameters(), self.model.q1.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        for target_param, param in zip(self.target_model.q2.parameters(), self.model.q2.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
