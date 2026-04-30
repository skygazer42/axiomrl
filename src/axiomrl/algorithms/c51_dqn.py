from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_c51_q_network import MLPC51QNetwork


def _c51_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    logits = batch["logits"]
    actions = batch["actions"].long()
    target_distributions = batch["target_distributions"]

    log_probs = F.log_softmax(logits, dim=-1)
    batch_size = int(actions.shape[0])
    num_atoms = int(log_probs.shape[-1])
    chosen_log_probs = log_probs.gather(1, actions.view(batch_size, 1, 1).expand(batch_size, 1, num_atoms)).squeeze(1)

    per_item_losses = -(target_distributions * chosen_log_probs).sum(dim=-1)
    weights = batch.get("weights")
    if weights is None:
        loss = per_item_losses.mean()
    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=per_item_losses.device)
        if weight_tensor.ndim != 1:
            weight_tensor = weight_tensor.reshape(-1)
        loss = (weight_tensor * per_item_losses).mean()

    return {"loss": loss}


def c51_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _c51_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _project_distribution(
    *,
    next_action_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    batch_size, num_atoms = next_action_probs.shape
    delta_z = (float(v_max) - float(v_min)) / float(num_atoms - 1)

    tz = rewards.unsqueeze(1) + gamma * (1.0 - dones).unsqueeze(1) * support.unsqueeze(0)
    tz = tz.clamp(float(v_min), float(v_max))
    b = (tz - float(v_min)) / delta_z

    lower = b.floor().long()
    upper = (lower + 1).clamp(max=num_atoms - 1)
    lower = lower.clamp(min=0, max=num_atoms - 1)

    upper_weight = b - lower.float()
    lower_weight = 1.0 - upper_weight
    same_mask = upper == lower
    lower_weight = torch.where(same_mask, torch.ones_like(lower_weight), lower_weight)
    upper_weight = torch.where(same_mask, torch.zeros_like(upper_weight), upper_weight)

    projected = torch.zeros_like(next_action_probs)
    offsets = torch.arange(batch_size, device=projected.device).unsqueeze(1) * num_atoms
    projected.view(-1).scatter_add_(0, (lower + offsets).view(-1), (next_action_probs * lower_weight).view(-1))
    projected.view(-1).scatter_add_(0, (upper + offsets).view(-1), (next_action_probs * upper_weight).view(-1))
    return projected


class C51DQN:
    def __init__(
        self,
        *,
        q_network: MLPC51QNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        v_min: float,
        v_max: float,
        num_atoms: int,
    ) -> None:
        if num_atoms != q_network.num_atoms:
            raise ValueError(f"expected num_atoms={q_network.num_atoms}, got {num_atoms}")
        if abs(float(v_min) - float(q_network.v_min)) > 1e-8 or abs(float(v_max) - float(q_network.v_max)) > 1e-8:
            raise ValueError("C51DQN v_min/v_max must match the q_network support range")

        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_atoms = int(num_atoms)
        self.last_td_errors: torch.Tensor | None = None

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        logits = self.q_network(obs)
        support = self.q_network.support

        with torch.no_grad():
            next_logits = self.target_network(next_obs)
            next_probs = F.softmax(next_logits, dim=-1)
            next_q_values = (next_probs * support).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)
            batch_size = int(next_actions.shape[0])
            next_action_probs = next_probs.gather(
                1, next_actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_atoms)
            ).squeeze(1)

            target_distributions = _project_distribution(
                next_action_probs=next_action_probs,
                rewards=rewards,
                dones=dones,
                support=support,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
            )

        loss_terms = _c51_loss_terms(
            {
                "logits": logits,
                "actions": actions,
                "target_distributions": target_distributions,
                "weights": batch.get("weights"),
            }
        )

        probs = F.softmax(logits, dim=-1)
        batch_size = int(actions.shape[0])
        chosen_probs = probs.gather(1, actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_atoms)).squeeze(1)
        q_values = (chosen_probs * support).sum(dim=-1)
        target_q_values = (target_distributions * support).sum(dim=-1)
        td_errors = target_q_values - q_values
        self.last_td_errors = td_errors.detach().abs()

        self.optimizer.zero_grad(set_to_none=True)
        loss_terms["loss"].backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {
            "loss": float(loss_terms["loss"].detach().cpu().item()),
            "q_value_mean": float(q_values.mean().detach().cpu().item()),
            "target_mean": float(target_q_values.mean().detach().cpu().item()),
            "td_error_mean": float(td_errors.abs().mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.q_network.train(True)
        self.target_network.train(False)

    def set_eval_mode(self) -> None:
        self.q_network.eval()
        self.target_network.eval()
