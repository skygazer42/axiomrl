from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_redq import MLPREDQModel


def _sample_target_critic_indices(
    *,
    num_critics: int,
    subset_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if subset_size < 1:
        raise ValueError(f"subset_size must be >= 1, got {subset_size}")
    if subset_size > num_critics:
        raise ValueError(f"subset_size must be <= num_critics, got {subset_size} > {num_critics}")
    return torch.randperm(int(num_critics), device=device)[: int(subset_size)]


def _redq_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    critic_q_values = torch.as_tensor(batch["critic_q_values"], dtype=torch.float32)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=critic_q_values.device)
    sampled_logprobs = torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32, device=critic_q_values.device)
    sampled_q_values = torch.as_tensor(batch["sampled_q_values"], dtype=torch.float32, device=critic_q_values.device)
    alpha = torch.as_tensor(batch["alpha"], dtype=torch.float32, device=critic_q_values.device)

    target_matrix = target_q_values.unsqueeze(1).expand_as(critic_q_values)
    per_value_loss = F.mse_loss(critic_q_values, target_matrix, reduction="none")
    per_sample_loss = per_value_loss.mean(dim=1)

    weights = batch.get("weights")
    if weights is None:
        critic_loss = per_sample_loss.mean()
    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=critic_q_values.device).reshape(-1)
        critic_loss = (per_sample_loss * weight_tensor).mean()

    actor_loss = (alpha * sampled_logprobs - sampled_q_values).mean()
    entropy_term = (alpha * sampled_logprobs).mean()

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "entropy_term": entropy_term,
    }


def redq_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _redq_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class REDQ:
    def __init__(
        self,
        *,
        model: MLPREDQModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        num_critics: int,
        subset_size: int,
    ) -> None:
        if int(num_critics) != int(model.num_critics):
            raise ValueError(f"expected num_critics={model.num_critics}, got {num_critics}")
        if int(subset_size) > int(num_critics):
            raise ValueError(f"subset_size must be <= num_critics, got {subset_size} > {num_critics}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=float(learning_rate))
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.num_critics = int(num_critics)
        self.subset_size = int(subset_size)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        current_q_values = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_policy = self.target_model.sample_actions(next_obs)
            target_q_ensemble = self.target_model.q_values(next_obs, next_policy.actions)
            subset_indices = _sample_target_critic_indices(
                num_critics=self.num_critics,
                subset_size=self.subset_size,
                device=target_q_ensemble.device,
            )
            target_subset = target_q_ensemble.index_select(dim=1, index=subset_indices)
            min_target_q = target_subset.min(dim=1).values
            target_q_values = rewards + self.gamma * (1.0 - dones) * (
                min_target_q - self.alpha * next_policy.logprobs
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _redq_loss_terms(
            {
                "critic_q_values": current_q_values,
                "target_q_values": target_q_values,
                "sampled_logprobs": torch.zeros_like(rewards),
                "sampled_q_values": torch.zeros_like(rewards),
                "alpha": self.alpha,
                "weights": batch.get("weights"),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled = self.model.sample_actions(obs)
        sampled_q_values = self.model.q_values(obs, sampled.actions).mean(dim=1)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _redq_loss_terms(
            {
                "critic_q_values": current_q_values.detach(),
                "target_q_values": target_q_values.detach(),
                "sampled_logprobs": sampled.logprobs,
                "sampled_q_values": sampled_q_values,
                "alpha": self.alpha,
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
