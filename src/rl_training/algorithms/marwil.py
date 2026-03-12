from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_iql import MLPIQLModel


def _marwil_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    value_predictions = torch.as_tensor(batch["value_predictions"], dtype=torch.float32)
    returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32, device=value_predictions.device)
    behavior_logprobs = torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32, device=value_predictions.device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=value_predictions.device)
    advantage_weights = torch.as_tensor(batch["advantage_weights"], dtype=torch.float32, device=value_predictions.device)
    advantage_norm_scale = torch.as_tensor(
        batch.get("advantage_norm_scale", 1.0),
        dtype=torch.float32,
        device=value_predictions.device,
    )
    vf_coeff = float(batch.get("vf_coeff", 1.0))

    value_loss = F.mse_loss(value_predictions, returns_to_go)
    scaled_value_loss = value_loss * vf_coeff
    actor_loss = -(advantage_weights * behavior_logprobs).mean()
    total_loss = actor_loss + scaled_value_loss

    return {
        "value_loss": value_loss,
        "scaled_value_loss": scaled_value_loss,
        "actor_loss": actor_loss,
        "total_loss": total_loss,
        "returns_to_go_mean": returns_to_go.mean(),
        "advantage_mean": advantages.mean(),
        "advantage_weight_mean": advantage_weights.mean(),
        "behavior_logprob_mean": behavior_logprobs.mean(),
        "advantage_norm_scale": advantage_norm_scale,
    }


def marwil_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _marwil_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class MARWIL:
    def __init__(
        self,
        *,
        model: MLPIQLModel,
        learning_rate: float,
        beta: float,
        vf_coeff: float,
        moving_average_sqd_adv_norm_start: float,
        moving_average_sqd_adv_norm_update_rate: float,
    ) -> None:
        if float(beta) < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if float(vf_coeff) < 0.0:
            raise ValueError(f"vf_coeff must be >= 0, got {vf_coeff}")
        if float(moving_average_sqd_adv_norm_start) <= 0.0:
            raise ValueError(
                "moving_average_sqd_adv_norm_start must be > 0, "
                f"got {moving_average_sqd_adv_norm_start}"
            )
        if not 0.0 < float(moving_average_sqd_adv_norm_update_rate) <= 1.0:
            raise ValueError(
                "moving_average_sqd_adv_norm_update_rate must be in (0, 1], "
                f"got {moving_average_sqd_adv_norm_update_rate}"
            )

        self.model = model
        self.policy = model
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate))
        self.value_optimizer = torch.optim.Adam(self.model.value_parameters(), lr=float(learning_rate))
        self.beta = float(beta)
        self.vf_coeff = float(vf_coeff)
        self.moving_average_sqd_adv_norm = float(moving_average_sqd_adv_norm_start)
        self.moving_average_sqd_adv_norm_update_rate = float(moving_average_sqd_adv_norm_update_rate)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32)

        value_predictions = self.model.value(obs)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_terms = _marwil_loss_terms(
            {
                "value_predictions": value_predictions,
                "returns_to_go": returns_to_go,
                "behavior_logprobs": torch.zeros_like(returns_to_go),
                "advantages": torch.zeros_like(returns_to_go),
                "advantage_weights": torch.ones_like(returns_to_go),
                "advantage_norm_scale": self.advantage_norm_scale,
                "vf_coeff": self.vf_coeff,
            }
        )
        value_terms["scaled_value_loss"].backward()
        self.value_optimizer.step()

        with torch.no_grad():
            advantages = returns_to_go - self.model.value(obs)
            batch_sqd_adv_norm = float(advantages.pow(2).mean().detach().cpu().item())
            self.moving_average_sqd_adv_norm = (
                (1.0 - self.moving_average_sqd_adv_norm_update_rate) * self.moving_average_sqd_adv_norm
                + self.moving_average_sqd_adv_norm_update_rate * batch_sqd_adv_norm
            )
            if self.beta == 0.0:
                advantage_weights = torch.ones_like(advantages)
            else:
                advantage_weights = torch.exp(self.beta * (advantages / self.advantage_norm_scale))

        behavior_logprobs = self.model.action_logprobs(obs, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _marwil_loss_terms(
            {
                "value_predictions": value_predictions.detach(),
                "returns_to_go": returns_to_go,
                "behavior_logprobs": behavior_logprobs,
                "advantages": advantages,
                "advantage_weights": advantage_weights,
                "advantage_norm_scale": self.advantage_norm_scale,
                "vf_coeff": self.vf_coeff,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        metrics = {
            "value_loss": float(value_terms["value_loss"].detach().cpu().item()),
            "scaled_value_loss": float(value_terms["scaled_value_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "total_loss": float(actor_terms["total_loss"].detach().cpu().item()),
            "returns_to_go_mean": float(actor_terms["returns_to_go_mean"].detach().cpu().item()),
            "advantage_mean": float(actor_terms["advantage_mean"].detach().cpu().item()),
            "advantage_weight_mean": float(actor_terms["advantage_weight_mean"].detach().cpu().item()),
            "behavior_logprob_mean": float(actor_terms["behavior_logprob_mean"].detach().cpu().item()),
            "advantage_norm_scale": float(actor_terms["advantage_norm_scale"].detach().cpu().item()),
            "moving_average_sqd_adv_norm": float(self.moving_average_sqd_adv_norm),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    @property
    def advantage_norm_scale(self) -> float:
        return float(max(self.moving_average_sqd_adv_norm, 1e-8) ** 0.5)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "moving_average_sqd_adv_norm": self.moving_average_sqd_adv_norm,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])
        self.moving_average_sqd_adv_norm = float(
            state_dict.get("moving_average_sqd_adv_norm", self.moving_average_sqd_adv_norm)
        )

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
