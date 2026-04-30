from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_bc import MLPBCModel


def bc_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    predictions = torch.as_tensor(batch["predictions"], dtype=torch.float32)
    targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=predictions.device)
    loss = F.mse_loss(predictions, targets)
    return {"bc_loss": float(loss.detach().cpu().item())}


class BC:
    def __init__(
        self,
        *,
        model: MLPBCModel,
        learning_rate: float,
    ) -> None:
        self.model = model
        self.policy = model
        self.optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        targets = torch.as_tensor(batch["actions"], dtype=torch.float32)
        predictions = self.model.actor(obs)
        loss = F.mse_loss(predictions, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        metrics = {
            "bc_loss": float(loss.detach().cpu().item()),
            "action_abs_mean": float(predictions.detach().abs().mean().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
