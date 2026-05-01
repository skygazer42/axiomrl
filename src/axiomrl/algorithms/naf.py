import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_naf import MLPNAFModel


def _naf_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    loss = F.mse_loss(batch["q_values"], batch["target_q_values"])
    return {
        "loss": loss,
        "q_value_mean": batch["q_values"].mean(),
        "target_q_mean": batch["target_q_values"].mean(),
    }


def naf_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _naf_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class NAF:
    def __init__(
        self,
        *,
        model: MLPNAFModel,
        learning_rate: float,
        gamma: float,
        tau: float,
    ) -> None:
        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.tau = tau

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.model.q_values(obs, actions)
        with torch.no_grad():
            next_state_values = self.target_model.state_values(next_obs)
            target_q_values = rewards + self.gamma * (1.0 - dones) * next_state_values

        terms = _naf_loss_terms(
            {
                "q_values": q_values,
                "target_q_values": target_q_values,
            }
        )

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        self.optimizer.step()
        self.soft_update_targets()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
