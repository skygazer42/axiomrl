from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_sac import MLPSACModel


def _sac_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"], batch["target_q_values"]
    )
    min_sampled_q = torch.minimum(batch["sampled_q1"], batch["sampled_q2"])
    entropy_term = batch["alpha"] * batch["sampled_logprobs"].mean()
    actor_loss = (batch["alpha"] * batch["sampled_logprobs"] - min_sampled_q).mean()

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "entropy_term": entropy_term,
    }


def sac_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "q1_values": torch.as_tensor(batch["q1_values"], dtype=torch.float32),
        "q2_values": torch.as_tensor(batch["q2_values"], dtype=torch.float32),
        "target_q_values": torch.as_tensor(batch["target_q_values"], dtype=torch.float32),
        "sampled_logprobs": torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32),
        "sampled_q1": torch.as_tensor(batch["sampled_q1"], dtype=torch.float32),
        "sampled_q2": torch.as_tensor(batch["sampled_q2"], dtype=torch.float32),
        "alpha": torch.as_tensor(batch["alpha"], dtype=torch.float32),
    }
    terms = _sac_loss_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class SAC:
    def __init__(
        self,
        *,
        model: MLPSACModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
    ) -> None:
        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=learning_rate, weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

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
            next_policy = self.target_model.sample_actions(next_obs)
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_policy.actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(target_q1, target_q2) - self.alpha * next_policy.logprobs
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _sac_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "sampled_logprobs": torch.zeros_like(target_q_values),
                "sampled_q1": torch.zeros_like(target_q_values),
                "sampled_q2": torch.zeros_like(target_q_values),
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=target_q_values.device),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled = self.model.sample_actions(obs)
        sampled_q1, sampled_q2 = self.model.q_values(obs, sampled.actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _sac_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "sampled_logprobs": sampled.logprobs,
                "sampled_q1": sampled_q1,
                "sampled_q2": sampled_q2,
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=sampled.logprobs.device),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "entropy_term": float(actor_terms["entropy_term"].detach().cpu().item()),
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
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
