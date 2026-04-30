from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_sac import MLPSACModel


def _cal_ql_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    sampled_logprobs = torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32, device=q1_values.device)
    sampled_q1 = torch.as_tensor(batch["sampled_q1"], dtype=torch.float32, device=q1_values.device)
    sampled_q2 = torch.as_tensor(batch["sampled_q2"], dtype=torch.float32, device=q1_values.device)
    calibrated_q1 = torch.as_tensor(batch["calibrated_q1"], dtype=torch.float32, device=q1_values.device)
    calibrated_q2 = torch.as_tensor(batch["calibrated_q2"], dtype=torch.float32, device=q1_values.device)
    returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32, device=q1_values.device)
    alpha = torch.as_tensor(batch["alpha"], dtype=torch.float32, device=q1_values.device)
    cql_penalty_q1 = torch.as_tensor(batch["cql_penalty_q1"], dtype=torch.float32, device=q1_values.device)
    cql_penalty_q2 = torch.as_tensor(batch["cql_penalty_q2"], dtype=torch.float32, device=q1_values.device)
    cql_alpha = torch.as_tensor(batch["cql_alpha"], dtype=torch.float32, device=q1_values.device)

    min_sampled_q = torch.minimum(sampled_q1, sampled_q2)
    entropy_term = alpha * sampled_logprobs.mean()
    actor_loss = (alpha * sampled_logprobs - min_sampled_q).mean()
    cql_penalty = cql_alpha * (cql_penalty_q1 + cql_penalty_q2)
    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    critic_loss = critic_loss + cql_penalty

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "entropy_term": entropy_term,
        "cql_penalty": cql_penalty,
        "calibrated_q1_mean": calibrated_q1.mean(),
        "calibrated_q2_mean": calibrated_q2.mean(),
        "returns_to_go_mean": returns_to_go.mean(),
    }


def cal_ql_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _cal_ql_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _repeat_observations(obs: torch.Tensor, num_samples: int) -> torch.Tensor:
    return obs.unsqueeze(1).expand(-1, num_samples, -1).reshape(obs.shape[0] * num_samples, obs.shape[1])


def _reshape_q_values(values: torch.Tensor, batch_size: int, num_samples: int) -> torch.Tensor:
    return values.reshape(batch_size, num_samples)


class CalQL:
    def __init__(
        self,
        *,
        model: MLPSACModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        cql_alpha: float,
        num_cql_samples: int,
    ) -> None:
        if float(cql_alpha) <= 0.0:
            raise ValueError(f"cql_alpha must be > 0, got {cql_alpha}")
        if int(num_cql_samples) < 1:
            raise ValueError(f"num_cql_samples must be >= 1, got {num_cql_samples}")

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
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.cql_alpha = float(cql_alpha)
        self.num_cql_samples = int(num_cql_samples)

    def _calibrated_cql_penalty(
        self,
        obs: torch.Tensor,
        current_q1: torch.Tensor,
        current_q2: torch.Tensor,
        returns_to_go: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, action_dim = current_q1.shape[0], self.model.action_dim
        device = obs.device

        repeated_obs = _repeat_observations(obs, self.num_cql_samples)
        random_actions = torch.empty(
            (batch_size * self.num_cql_samples, action_dim),
            dtype=torch.float32,
            device=device,
        ).uniform_(-1.0, 1.0)
        random_q1, random_q2 = self.model.q_values(repeated_obs, random_actions)

        with torch.no_grad():
            sampled_actions = self.model.sample_actions(repeated_obs).actions
        sampled_q1, sampled_q2 = self.model.q_values(repeated_obs, sampled_actions)

        random_q1 = _reshape_q_values(random_q1, batch_size, self.num_cql_samples)
        random_q2 = _reshape_q_values(random_q2, batch_size, self.num_cql_samples)
        sampled_q1 = _reshape_q_values(sampled_q1, batch_size, self.num_cql_samples)
        sampled_q2 = _reshape_q_values(sampled_q2, batch_size, self.num_cql_samples)
        repeated_returns_to_go = returns_to_go.reshape(batch_size, 1).expand(-1, self.num_cql_samples)
        calibrated_q1 = torch.maximum(sampled_q1, repeated_returns_to_go)
        calibrated_q2 = torch.maximum(sampled_q2, repeated_returns_to_go)

        conservative_q1 = torch.logsumexp(torch.cat([random_q1, calibrated_q1], dim=1), dim=1).mean()
        conservative_q2 = torch.logsumexp(torch.cat([random_q2, calibrated_q2], dim=1), dim=1).mean()
        return (
            conservative_q1 - current_q1.mean(),
            conservative_q2 - current_q2.mean(),
            calibrated_q1.mean(),
            calibrated_q2.mean(),
        )

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
        returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32, device=obs.device)

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_policy = self.target_model.sample_actions(next_obs)
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_policy.actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(target_q1, target_q2) - self.alpha * next_policy.logprobs
            )

        cql_penalty_q1, cql_penalty_q2, calibrated_q1_mean, calibrated_q2_mean = self._calibrated_cql_penalty(
            obs,
            current_q1,
            current_q2,
            returns_to_go,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _cal_ql_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "sampled_logprobs": torch.zeros_like(target_q_values),
                "sampled_q1": torch.zeros_like(target_q_values),
                "sampled_q2": torch.zeros_like(target_q_values),
                "calibrated_q1": calibrated_q1_mean.reshape(1),
                "calibrated_q2": calibrated_q2_mean.reshape(1),
                "returns_to_go": returns_to_go,
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=target_q_values.device),
                "cql_penalty_q1": cql_penalty_q1,
                "cql_penalty_q2": cql_penalty_q2,
                "cql_alpha": torch.as_tensor(self.cql_alpha, dtype=torch.float32, device=target_q_values.device),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled = self.model.sample_actions(obs)
        sampled_q1, sampled_q2 = self.model.q_values(obs, sampled.actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _cal_ql_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "sampled_logprobs": sampled.logprobs,
                "sampled_q1": sampled_q1,
                "sampled_q2": sampled_q2,
                "calibrated_q1": critic_terms["calibrated_q1_mean"].detach().reshape(1),
                "calibrated_q2": critic_terms["calibrated_q2_mean"].detach().reshape(1),
                "returns_to_go": returns_to_go,
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=sampled.logprobs.device),
                "cql_penalty_q1": critic_terms["cql_penalty"].detach().new_zeros(()),
                "cql_penalty_q2": critic_terms["cql_penalty"].detach().new_zeros(()),
                "cql_alpha": torch.as_tensor(self.cql_alpha, dtype=torch.float32, device=sampled.logprobs.device),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "entropy_term": float(actor_terms["entropy_term"].detach().cpu().item()),
            "cql_penalty": float(critic_terms["cql_penalty"].detach().cpu().item()),
            "calibrated_q1_mean": float(critic_terms["calibrated_q1_mean"].detach().cpu().item()),
            "calibrated_q2_mean": float(critic_terms["calibrated_q2_mean"].detach().cpu().item()),
            "returns_to_go_mean": float(critic_terms["returns_to_go_mean"].detach().cpu().item()),
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
