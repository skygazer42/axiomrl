from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms._advantage_utils import normalize_advantages
from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_iql import MLPIQLModel


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return normalize_advantages(advantages)


def _awr_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    value_predictions = torch.as_tensor(batch["value_predictions"], dtype=torch.float32)
    returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32, device=value_predictions.device)
    behavior_logprobs = torch.as_tensor(
        batch["behavior_logprobs"], dtype=torch.float32, device=value_predictions.device
    )
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=value_predictions.device)
    advantage_weights = torch.as_tensor(
        batch["advantage_weights"], dtype=torch.float32, device=value_predictions.device
    )

    value_loss = F.mse_loss(value_predictions, returns_to_go)
    actor_loss = -(advantage_weights * behavior_logprobs).mean()

    return {
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "returns_to_go_mean": returns_to_go.mean(),
        "advantage_mean": advantages.mean(),
        "advantage_weight_mean": advantage_weights.mean(),
        "behavior_logprob_mean": behavior_logprobs.mean(),
    }


def awr_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _awr_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class AWR:
    def __init__(
        self,
        *,
        model: MLPIQLModel,
        learning_rate: float,
        beta: float,
        max_weight: float,
        normalize_advantages: bool = True,
    ) -> None:
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if float(max_weight) <= 0.0:
            raise ValueError(f"max_weight must be > 0, got {max_weight}")

        self.model = model
        self.policy = model
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.value_optimizer = torch.optim.Adam(
            self.model.value_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.beta = float(beta)
        self.max_weight = float(max_weight)
        self.normalize_advantages = bool(normalize_advantages)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32)

        value_predictions = self.model.value(obs)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_terms = _awr_loss_terms(
            {
                "value_predictions": value_predictions,
                "returns_to_go": returns_to_go,
                "behavior_logprobs": torch.zeros_like(returns_to_go),
                "advantages": torch.zeros_like(returns_to_go),
                "advantage_weights": torch.ones_like(returns_to_go),
            }
        )
        value_terms["value_loss"].backward()
        self.value_optimizer.step()

        with torch.no_grad():
            advantages = returns_to_go - self.model.value(obs)
            if self.normalize_advantages:
                advantages = _normalize_advantages(advantages)
            advantage_weights = torch.exp(advantages / self.beta).clamp(max=self.max_weight)

        behavior_logprobs = self.model.action_logprobs(obs, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _awr_loss_terms(
            {
                "value_predictions": value_predictions.detach(),
                "returns_to_go": returns_to_go,
                "behavior_logprobs": behavior_logprobs,
                "advantages": advantages,
                "advantage_weights": advantage_weights,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        metrics = {
            "value_loss": float(value_terms["value_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "returns_to_go_mean": float(actor_terms["returns_to_go_mean"].detach().cpu().item()),
            "advantage_mean": float(actor_terms["advantage_mean"].detach().cpu().item()),
            "advantage_weight_mean": float(actor_terms["advantage_weight_mean"].detach().cpu().item()),
            "behavior_logprob_mean": float(actor_terms["behavior_logprob_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
