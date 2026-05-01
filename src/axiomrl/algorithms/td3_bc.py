import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_td3 import MLPTD3Model


def _td3_bc_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"], batch["target_q_values"]
    )
    actor_loss = batch["bc_loss"] - batch["bc_lambda"] * batch["actor_q_values"].mean()
    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "bc_loss": batch["bc_loss"],
        "bc_lambda": batch["bc_lambda"],
    }


def td3_bc_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _td3_bc_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class TD3BC:
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
        bc_alpha: float,
    ) -> None:
        if int(policy_delay) < 1:
            raise ValueError(f"policy_delay must be >= 1, got {policy_delay}")
        if float(bc_alpha) <= 0.0:
            raise ValueError(f"bc_alpha must be > 0, got {bc_alpha}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=learning_rate, weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)
        self.bc_alpha = float(bc_alpha)
        self.update_count = 0

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
            next_actions = self.target_model.actor(next_obs)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_model.q_values(next_obs, next_actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _td3_bc_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "actor_q_values": current_q1.detach(),
                "bc_loss": torch.zeros((), dtype=torch.float32, device=obs.device),
                "bc_lambda": torch.zeros((), dtype=torch.float32, device=obs.device),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        actor_loss_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        bc_loss_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        bc_lambda_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        self.update_count += 1

        if self.update_count % self.policy_delay == 0:
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_actions = self.model.actor(obs)
            actor_q1, _ = self.model.q_values(obs, policy_actions)
            bc_loss = F.mse_loss(policy_actions, actions)
            q_scale = actor_q1.abs().mean().detach().clamp(min=1e-6)
            bc_lambda = torch.as_tensor(self.bc_alpha, dtype=torch.float32, device=obs.device) / q_scale
            actor_terms = _td3_bc_loss_terms(
                {
                    "q1_values": current_q1.detach(),
                    "q2_values": current_q2.detach(),
                    "target_q_values": target_q_values.detach(),
                    "actor_q_values": actor_q1,
                    "bc_loss": bc_loss,
                    "bc_lambda": bc_lambda,
                }
            )
            actor_terms["actor_loss"].backward()
            self.actor_optimizer.step()
            self.soft_update_targets()
            actor_loss_value = actor_terms["actor_loss"].detach()
            bc_loss_value = actor_terms["bc_loss"].detach()
            bc_lambda_value = actor_terms["bc_lambda"].detach()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_loss_value.cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "bc_loss": float(bc_loss_value.cpu().item()),
            "bc_lambda": float(bc_lambda_value.cpu().item()),
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
