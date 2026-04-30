from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_redq import MLPREDQModel


def critic_diversity_loss(action_gradients: torch.Tensor | float) -> torch.Tensor:
    gradients = torch.as_tensor(action_gradients, dtype=torch.float32)
    if gradients.ndim != 3:
        raise ValueError(
            f"expected action_gradients to have shape [batch, num_critics, action_dim], got {tuple(gradients.shape)}"
        )
    num_critics = int(gradients.shape[1])
    if num_critics < 2:
        raise ValueError(f"critic_diversity_loss requires at least 2 critics, got {num_critics}")

    normalized = gradients / (torch.norm(gradients, p=2, dim=2, keepdim=True) + 1e-10)
    pairwise = normalized @ normalized.transpose(1, 2)
    masks = torch.eye(num_critics, device=normalized.device, dtype=normalized.dtype).unsqueeze(0)
    loss = ((1.0 - masks) * pairwise).sum(dim=(1, 2)).mean()
    return loss / float(num_critics - 1)


def _edac_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    critic_q_values = torch.as_tensor(batch["critic_q_values"], dtype=torch.float32)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=critic_q_values.device)
    sampled_logprobs = torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32, device=critic_q_values.device)
    sampled_q_values = torch.as_tensor(batch["sampled_q_values"], dtype=torch.float32, device=critic_q_values.device)
    alpha = torch.as_tensor(batch["alpha"], dtype=torch.float32, device=critic_q_values.device)
    diversity = torch.as_tensor(batch["diversity_loss"], dtype=torch.float32, device=critic_q_values.device)
    eta = torch.as_tensor(batch["eta"], dtype=torch.float32, device=critic_q_values.device)

    if critic_q_values.ndim == 1:
        target_matrix = target_q_values.reshape_as(critic_q_values)
        critic_mse_loss = F.mse_loss(critic_q_values, target_matrix)
        q_data_std = torch.zeros((), dtype=torch.float32, device=critic_q_values.device)
    else:
        target_matrix = target_q_values.unsqueeze(1).expand_as(critic_q_values)
        critic_mse_loss = F.mse_loss(critic_q_values, target_matrix, reduction="none").mean(dim=0).sum()
        q_data_std = critic_q_values.std(dim=1).mean()

    if sampled_q_values.ndim == 1:
        sampled_q_min = sampled_q_values
        q_policy_std = torch.zeros((), dtype=torch.float32, device=critic_q_values.device)
    else:
        sampled_q_min = sampled_q_values.min(dim=1).values
        q_policy_std = sampled_q_values.std(dim=1).mean()

    critic_loss = critic_mse_loss + eta * diversity
    actor_loss = (alpha * sampled_logprobs - sampled_q_min).mean()
    entropy_term = (alpha * sampled_logprobs).mean()

    return {
        "critic_loss": critic_loss,
        "critic_mse_loss": critic_mse_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "entropy_term": entropy_term,
        "diversity_loss": diversity,
        "q_data_std": q_data_std,
        "q_policy_std": q_policy_std,
    }


def edac_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _edac_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class EDAC:
    def __init__(
        self,
        *,
        model: MLPREDQModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        num_critics: int,
        eta: float,
    ) -> None:
        if int(num_critics) != int(model.num_critics):
            raise ValueError(f"expected num_critics={model.num_critics}, got {num_critics}")
        if int(num_critics) < 2:
            raise ValueError(f"num_critics must be >= 2, got {num_critics}")
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if float(eta) < 0.0:
            raise ValueError(f"eta must be >= 0, got {eta}")

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
        self.num_critics = int(num_critics)
        self.eta = float(eta)

    def _critic_action_gradients(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        gradients: list[torch.Tensor] = []
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        for critic in self.model.critics:
            critic_actions = actions.detach().clone().requires_grad_(True)
            critic_inputs = torch.cat([obs_tensor, critic_actions], dim=-1)
            q_values = critic(critic_inputs).squeeze(-1)
            action_grad = torch.autograd.grad(q_values.sum(), critic_actions, create_graph=True)[0]
            gradients.append(action_grad)

        return torch.stack(gradients, dim=1)

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
            min_target_q = target_q_ensemble.min(dim=1).values
            target_q_values = rewards + self.gamma * (1.0 - dones) * (min_target_q - self.alpha * next_policy.logprobs)

        diversity = (
            critic_diversity_loss(self._critic_action_gradients(obs, actions))
            if self.eta > 0.0
            else torch.zeros(
                (),
                dtype=torch.float32,
                device=obs.device,
            )
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _edac_loss_terms(
            {
                "critic_q_values": current_q_values,
                "target_q_values": target_q_values,
                "sampled_logprobs": torch.zeros_like(rewards),
                "sampled_q_values": torch.zeros_like(current_q_values),
                "alpha": self.alpha,
                "diversity_loss": diversity,
                "eta": self.eta,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled = self.model.sample_actions(obs)
        sampled_q_values = self.model.q_values(obs, sampled.actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _edac_loss_terms(
            {
                "critic_q_values": current_q_values.detach(),
                "target_q_values": target_q_values.detach(),
                "sampled_logprobs": sampled.logprobs,
                "sampled_q_values": sampled_q_values,
                "alpha": self.alpha,
                "diversity_loss": torch.zeros((), dtype=torch.float32, device=sampled.logprobs.device),
                "eta": self.eta,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "critic_mse_loss": float(critic_terms["critic_mse_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "entropy_term": float(actor_terms["entropy_term"].detach().cpu().item()),
            "diversity_loss": float(critic_terms["diversity_loss"].detach().cpu().item()),
            "q_data_std": float(critic_terms["q_data_std"].detach().cpu().item()),
            "q_policy_std": float(actor_terms["q_policy_std"].detach().cpu().item()),
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
