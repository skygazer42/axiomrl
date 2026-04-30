from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_bear import MLPBEARModel


def _bear_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    reconstruction_loss = torch.as_tensor(batch["reconstruction_loss"], dtype=torch.float32, device=q1_values.device)
    kl_loss = torch.as_tensor(batch["kl_loss"], dtype=torch.float32, device=q1_values.device)
    behavior_kl_weight = torch.as_tensor(batch["behavior_kl_weight"], dtype=torch.float32, device=q1_values.device)
    actor_q_values = torch.as_tensor(batch["actor_q_values"], dtype=torch.float32, device=q1_values.device)
    mmd_loss = torch.as_tensor(batch["mmd_loss"], dtype=torch.float32, device=q1_values.device)
    mmd_alpha = torch.as_tensor(batch["mmd_alpha"], dtype=torch.float32, device=q1_values.device)

    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    actor_loss = mmd_alpha * mmd_loss - actor_q_values.mean()
    behavior_loss = reconstruction_loss + behavior_kl_weight * kl_loss
    return {
        "behavior_loss": behavior_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "mmd_loss": mmd_loss,
        "target_q_mean": target_q_values.mean(),
    }


def bear_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _bear_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _repeat_observations(obs: torch.Tensor, count: int) -> torch.Tensor:
    return obs.unsqueeze(1).expand(-1, count, -1).reshape(obs.shape[0] * count, obs.shape[1])


def _gaussian_mmd(x: torch.Tensor, y: torch.Tensor, *, sigma: float) -> torch.Tensor:
    scale = 2.0 * float(sigma) * float(sigma)

    def _kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        squared_distance = (a.unsqueeze(2) - b.unsqueeze(1)).pow(2).sum(dim=-1)
        return torch.exp(-squared_distance / scale)

    k_xx = _kernel(x, x)
    k_yy = _kernel(y, y)
    k_xy = _kernel(x, y)
    return (k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2.0 * k_xy.mean(dim=(1, 2))).clamp(min=0.0).mean()


class BEAR:
    def __init__(
        self,
        *,
        model: MLPBEARModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        behavior_kl_weight: float,
        mmd_sigma: float,
        mmd_alpha: float,
        num_mmd_action_samples: int,
    ) -> None:
        if float(behavior_kl_weight) < 0.0:
            raise ValueError(f"behavior_kl_weight must be >= 0, got {behavior_kl_weight}")
        if float(mmd_sigma) <= 0.0:
            raise ValueError(f"mmd_sigma must be > 0, got {mmd_sigma}")
        if float(mmd_alpha) <= 0.0:
            raise ValueError(f"mmd_alpha must be > 0, got {mmd_alpha}")
        if int(num_mmd_action_samples) < 1:
            raise ValueError(f"num_mmd_action_samples must be >= 1, got {num_mmd_action_samples}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.behavior_optimizer = torch.optim.Adam(self.model.behavior_parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.behavior_kl_weight = float(behavior_kl_weight)
        self.mmd_sigma = float(mmd_sigma)
        self.mmd_alpha = float(mmd_alpha)
        self.num_mmd_action_samples = int(num_mmd_action_samples)

    def _mmd_loss(self, obs: torch.Tensor) -> torch.Tensor:
        behavior_actions = self.model.sample_behavior_actions(
            obs,
            num_action_samples=self.num_mmd_action_samples,
            deterministic=False,
        )
        repeated_obs = _repeat_observations(obs, self.num_mmd_action_samples)
        policy_actions = self.model.sample_actions(repeated_obs, deterministic=False).actions.reshape(
            obs.shape[0],
            self.num_mmd_action_samples,
            self.model.action_dim,
        )
        return _gaussian_mmd(policy_actions, behavior_actions.detach(), sigma=self.mmd_sigma)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        reconstructed_actions, latent_mean, latent_log_std = self.model.reconstruct_behavior(obs, actions)
        reconstruction_loss = F.mse_loss(reconstructed_actions, actions)
        latent_variance = torch.exp(2.0 * latent_log_std)
        kl_loss = (
            -0.5 * (1.0 + 2.0 * latent_log_std - latent_mean.pow(2) - latent_variance).sum(dim=-1).mean()
        )

        self.behavior_optimizer.zero_grad(set_to_none=True)
        behavior_terms = _bear_loss_terms(
            {
                "q1_values": torch.zeros_like(rewards),
                "q2_values": torch.zeros_like(rewards),
                "target_q_values": torch.zeros_like(rewards),
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "behavior_kl_weight": torch.as_tensor(self.behavior_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": torch.zeros_like(rewards),
                "mmd_loss": torch.zeros((), dtype=torch.float32, device=obs.device),
                "mmd_alpha": torch.as_tensor(self.mmd_alpha, dtype=torch.float32, device=obs.device),
            }
        )
        behavior_terms["behavior_loss"].backward()
        self.behavior_optimizer.step()

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_policy = self.target_model.sample_actions(next_obs, deterministic=False)
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_policy.actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _bear_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "reconstruction_loss": reconstruction_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "behavior_kl_weight": torch.as_tensor(self.behavior_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": torch.zeros_like(rewards),
                "mmd_loss": torch.zeros((), dtype=torch.float32, device=obs.device),
                "mmd_alpha": torch.as_tensor(self.mmd_alpha, dtype=torch.float32, device=obs.device),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled_actions = self.model.sample_actions(obs, deterministic=False).actions
        actor_q1, actor_q2 = self.model.q_values(obs, sampled_actions)
        mmd_loss = self._mmd_loss(obs)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _bear_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "behavior_kl_weight": torch.as_tensor(self.behavior_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": torch.minimum(actor_q1, actor_q2),
                "mmd_loss": mmd_loss,
                "mmd_alpha": torch.as_tensor(self.mmd_alpha, dtype=torch.float32, device=obs.device),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "behavior_loss": float(behavior_terms["behavior_loss"].detach().cpu().item()),
            "reconstruction_loss": float(behavior_terms["reconstruction_loss"].detach().cpu().item()),
            "kl_loss": float(behavior_terms["kl_loss"].detach().cpu().item()),
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "mmd_loss": float(actor_terms["mmd_loss"].detach().cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "behavior_optimizer": self.behavior_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.behavior_optimizer.load_state_dict(state_dict["behavior_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
