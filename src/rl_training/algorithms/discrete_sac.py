from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_discrete_sac import MLPDiscreteSACModel


def _discrete_sac_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"],
        batch["target_q_values"],
    )
    min_policy_q = torch.minimum(batch["policy_q1"], batch["policy_q2"])
    actor_loss = (batch["action_probs"] * (batch["alpha"] * batch["log_action_probs"] - min_policy_q)).sum(dim=-1).mean()
    entropy = -(batch["action_probs"] * batch["log_action_probs"]).sum(dim=-1).mean()

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "entropy": entropy,
    }


def discrete_sac_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "q1_values": torch.as_tensor(batch["q1_values"], dtype=torch.float32),
        "q2_values": torch.as_tensor(batch["q2_values"], dtype=torch.float32),
        "target_q_values": torch.as_tensor(batch["target_q_values"], dtype=torch.float32),
        "action_probs": torch.as_tensor(batch["action_probs"], dtype=torch.float32),
        "log_action_probs": torch.as_tensor(batch["log_action_probs"], dtype=torch.float32),
        "policy_q1": torch.as_tensor(batch["policy_q1"], dtype=torch.float32),
        "policy_q2": torch.as_tensor(batch["policy_q2"], dtype=torch.float32),
        "alpha": torch.as_tensor(batch["alpha"], dtype=torch.float32),
    }
    terms = _discrete_sac_loss_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class DiscreteSAC:
    def __init__(
        self,
        *,
        model: MLPDiscreteSACModel,
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
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        current_q1_all, current_q2_all = self.model.q_values(obs)
        current_q1 = current_q1_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        current_q2 = current_q2_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_action_probs, next_log_action_probs = self.target_model.policy(next_obs)
            target_q1_all, target_q2_all = self.target_model.q_values(next_obs)
            target_state_values = (
                next_action_probs
                * (torch.minimum(target_q1_all, target_q2_all) - self.alpha * next_log_action_probs)
            ).sum(dim=-1)
            target_q_values = rewards + self.gamma * (1.0 - dones) * target_state_values

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = F.mse_loss(current_q1, target_q_values) + F.mse_loss(current_q2, target_q_values)
        critic_loss.backward()
        self.critic_optimizer.step()

        action_probs, log_action_probs = self.model.policy(obs)
        policy_q1, policy_q2 = self.model.q_values(obs)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _discrete_sac_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "action_probs": action_probs,
                "log_action_probs": log_action_probs,
                "policy_q1": policy_q1,
                "policy_q2": policy_q2,
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=action_probs.device),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "entropy": float(actor_terms["entropy"].detach().cpu().item()),
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

