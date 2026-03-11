from __future__ import annotations

import copy
from typing import Any

import torch

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_tqc import MLPTQCModel


def _quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    *,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    if pred_quantiles.ndim != 2 or target_quantiles.ndim != 2:
        raise ValueError("expected pred_quantiles and target_quantiles to have shape (batch, quantiles)")

    td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_errors = td_errors.abs()

    huber = torch.where(
        abs_errors <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (abs_errors - 0.5 * kappa),
    )

    indicator = (td_errors.detach() < 0).to(dtype=pred_quantiles.dtype)
    tau = taus.view(1, -1, 1).to(device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    quantile_weights = (tau - indicator).abs()
    return (quantile_weights * huber).mean(dim=(1, 2))


def _truncate_quantiles(quantiles: torch.Tensor, *, top_quantiles_to_drop_per_net: int) -> torch.Tensor:
    if quantiles.ndim != 3:
        raise ValueError("expected quantiles to have shape (batch, num_critics, num_quantiles)")

    batch_size, num_critics, num_quantiles = quantiles.shape
    drop_count = int(top_quantiles_to_drop_per_net) * int(num_critics)
    total_quantiles = int(num_critics * num_quantiles)
    if drop_count < 0:
        raise ValueError(f"top_quantiles_to_drop_per_net must be >= 0, got {top_quantiles_to_drop_per_net}")
    if drop_count >= total_quantiles:
        raise ValueError("cannot drop all target quantiles")

    flattened = quantiles.reshape(batch_size, total_quantiles)
    sorted_quantiles, _ = torch.sort(flattened, dim=-1)
    keep_count = total_quantiles - drop_count
    return sorted_quantiles[:, :keep_count]


def _tqc_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    critic_quantiles = torch.as_tensor(batch["critic_quantiles"], dtype=torch.float32)
    target_quantiles = torch.as_tensor(batch["target_quantiles"], dtype=torch.float32, device=critic_quantiles.device)
    taus = torch.as_tensor(batch["taus"], dtype=torch.float32, device=critic_quantiles.device)
    alpha = torch.as_tensor(batch["alpha"], dtype=torch.float32, device=critic_quantiles.device)
    sampled_logprobs = torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32, device=critic_quantiles.device)
    sampled_q_values = torch.as_tensor(batch["sampled_q_values"], dtype=torch.float32, device=critic_quantiles.device)
    kappa = float(batch.get("kappa", 1.0))

    per_critic_losses = []
    for critic_index in range(int(critic_quantiles.shape[1])):
        per_critic_losses.append(
            _quantile_huber_loss(
                critic_quantiles[:, critic_index, :],
                target_quantiles,
                taus=taus,
                kappa=kappa,
            )
        )
    stacked_losses = torch.stack(per_critic_losses, dim=1)
    weights = batch.get("weights")
    if weights is None:
        critic_loss = stacked_losses.mean()
    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=stacked_losses.device)
        if weight_tensor.ndim != 1:
            weight_tensor = weight_tensor.reshape(-1)
        critic_loss = (stacked_losses.mean(dim=1) * weight_tensor).mean()

    actor_loss = (alpha * sampled_logprobs - sampled_q_values).mean()
    entropy_term = (alpha * sampled_logprobs).mean()

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_quantiles.mean(),
        "entropy_term": entropy_term,
    }


def tqc_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _tqc_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class TQC:
    def __init__(
        self,
        *,
        model: MLPTQCModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        top_quantiles_to_drop_per_net: int,
        num_quantiles: int,
        kappa: float = 1.0,
    ) -> None:
        if int(num_quantiles) != int(model.num_quantiles):
            raise ValueError(f"expected num_quantiles={model.num_quantiles}, got {num_quantiles}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=float(learning_rate))
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.top_quantiles_to_drop_per_net = int(top_quantiles_to_drop_per_net)
        self.num_quantiles = int(num_quantiles)
        self.kappa = float(kappa)
        self.taus = (torch.arange(self.num_quantiles, dtype=torch.float32) + 0.5) / float(self.num_quantiles)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        current_quantiles = self.model.quantile_values(obs, actions)

        with torch.no_grad():
            next_policy = self.target_model.sample_actions(next_obs)
            next_quantiles = self.target_model.quantile_values(next_obs, next_policy.actions)
            truncated_next_quantiles = _truncate_quantiles(
                next_quantiles,
                top_quantiles_to_drop_per_net=self.top_quantiles_to_drop_per_net,
            )
            target_quantiles = rewards.unsqueeze(1) + self.gamma * (1.0 - dones).unsqueeze(1) * (
                truncated_next_quantiles - self.alpha * next_policy.logprobs.unsqueeze(1)
            )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _tqc_loss_terms(
            {
                "critic_quantiles": current_quantiles,
                "target_quantiles": target_quantiles,
                "taus": self.taus.to(device=current_quantiles.device),
                "sampled_logprobs": torch.zeros_like(rewards),
                "sampled_q_values": torch.zeros_like(rewards),
                "alpha": self.alpha,
                "kappa": self.kappa,
                "weights": batch.get("weights"),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled = self.model.sample_actions(obs)
        sampled_q_values = self.model.expected_q_values(obs, sampled.actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _tqc_loss_terms(
            {
                "critic_quantiles": current_quantiles.detach(),
                "target_quantiles": target_quantiles.detach(),
                "taus": self.taus.to(device=current_quantiles.device),
                "sampled_logprobs": sampled.logprobs,
                "sampled_q_values": sampled_q_values,
                "alpha": self.alpha,
                "kappa": self.kappa,
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
