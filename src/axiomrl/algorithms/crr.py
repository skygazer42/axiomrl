from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_sac import MLPSACModel


def _crr_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"], batch["target_q_values"]
    )
    actor_loss = -(batch["advantage_weights"] * batch["behavior_logprobs"]).mean()
    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "advantage_mean": batch["advantages"].mean(),
        "advantage_weight_mean": batch["advantage_weights"].mean(),
        "behavior_logprob_mean": batch["behavior_logprobs"].mean(),
    }


def crr_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    behavior_logprobs = torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32, device=q1_values.device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=q1_values.device)
    advantage_weights = torch.as_tensor(batch["advantage_weights"], dtype=torch.float32, device=q1_values.device)
    terms = _crr_loss_terms(
        {
            "q1_values": q1_values,
            "q2_values": q2_values,
            "target_q_values": target_q_values,
            "behavior_logprobs": behavior_logprobs,
            "advantages": advantages,
            "advantage_weights": advantage_weights,
        }
    )
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _repeat_observations(obs: torch.Tensor, num_samples: int) -> torch.Tensor:
    return obs.unsqueeze(1).expand(-1, num_samples, -1).reshape(obs.shape[0] * num_samples, obs.shape[1])


class CRR:
    def __init__(
        self,
        *,
        model: MLPSACModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        beta: float,
        n_action_samples: int,
        max_weight: float,
        advantage_type: str,
        weight_type: str,
    ) -> None:
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if int(n_action_samples) < 1:
            raise ValueError(f"n_action_samples must be >= 1, got {n_action_samples}")
        if float(max_weight) <= 0.0:
            raise ValueError(f"max_weight must be > 0, got {max_weight}")
        if advantage_type not in {"mean", "max"}:
            raise ValueError(f"advantage_type must be 'mean' or 'max', got {advantage_type!r}")
        if weight_type not in {"binary", "exp"}:
            raise ValueError(f"weight_type must be 'binary' or 'exp', got {weight_type!r}")

        # This v1 stays close to the package's existing offline actor-critic
        # path so CRR can share evaluation, checkpointing, and dataset tooling.
        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.beta = float(beta)
        self.n_action_samples = int(n_action_samples)
        self.max_weight = float(max_weight)
        self.advantage_type = advantage_type
        self.weight_type = weight_type

    def _policy_values(self, obs: torch.Tensor) -> torch.Tensor:
        repeated_obs = _repeat_observations(obs, self.n_action_samples)
        sampled_actions = self.model.sample_actions(repeated_obs).actions
        sampled_q1, sampled_q2 = self.model.q_values(repeated_obs, sampled_actions)
        sampled_q = torch.minimum(sampled_q1, sampled_q2).reshape(obs.shape[0], self.n_action_samples)
        if self.advantage_type == "max":
            return sampled_q.max(dim=1).values
        return sampled_q.mean(dim=1)

    def _advantage_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        if self.weight_type == "binary":
            return (advantages > 0.0).to(dtype=torch.float32)
        return torch.exp(advantages / self.beta).clamp(max=self.max_weight)

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
            next_actions = self.target_model.sample_actions(next_obs).actions
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _crr_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "behavior_logprobs": torch.zeros_like(target_q_values),
                "advantages": torch.zeros_like(target_q_values),
                "advantage_weights": torch.ones_like(target_q_values),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            behavior_q1, behavior_q2 = self.model.q_values(obs, actions)
            behavior_q = torch.minimum(behavior_q1, behavior_q2)
            policy_values = self._policy_values(obs)
            advantages = behavior_q - policy_values
            advantage_weights = self._advantage_weights(advantages)

        behavior_logprobs = self.model.action_logprobs(obs, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _crr_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "behavior_logprobs": behavior_logprobs,
                "advantages": advantages,
                "advantage_weights": advantage_weights,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "advantage_mean": float(actor_terms["advantage_mean"].detach().cpu().item()),
            "advantage_weight_mean": float(actor_terms["advantage_weight_mean"].detach().cpu().item()),
            "behavior_logprob_mean": float(actor_terms["behavior_logprob_mean"].detach().cpu().item()),
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
