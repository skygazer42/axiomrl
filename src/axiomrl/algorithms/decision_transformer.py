from typing import Any

import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.decision_transformer import DecisionTransformerModel


def _masked_action_mse(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_token_mse = (predictions - targets).pow(2).mean(dim=-1)
    valid_tokens = mask.to(dtype=torch.float32, device=per_token_mse.device)
    masked_tokens = valid_tokens.sum()
    denominator = masked_tokens.clamp(min=1.0)
    action_mse = (per_token_mse * valid_tokens).sum() / denominator
    return action_mse, action_mse, masked_tokens


def decision_transformer_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    predictions = torch.as_tensor(batch["predictions"], dtype=torch.float32)
    targets = torch.as_tensor(batch["targets"], dtype=torch.float32, device=predictions.device)
    mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=predictions.device)
    loss, action_mse, masked_tokens = _masked_action_mse(
        predictions=predictions,
        targets=targets,
        mask=mask,
    )
    return {
        "decision_transformer_loss": float(loss.detach().cpu().item()),
        "action_mse": float(action_mse.detach().cpu().item()),
        "masked_tokens": float(masked_tokens.detach().cpu().item()),
    }


class DecisionTransformer:
    def __init__(
        self,
        *,
        model: DecisionTransformerModel,
        learning_rate: float,
    ) -> None:
        self.model = model
        self.policy = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), weight_decay=0.0)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        device = next(self.model.parameters()).device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
        returns_to_go = torch.as_tensor(batch["returns_to_go"], dtype=torch.float32, device=device)
        timesteps = torch.as_tensor(batch["timesteps"], dtype=torch.int64, device=device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)

        predictions = self.model.predict_actions(
            obs=obs,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            mask=mask,
        )
        loss, action_mse, masked_tokens = _masked_action_mse(
            predictions=predictions,
            targets=actions,
            mask=mask,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        metrics = {
            "decision_transformer_loss": float(loss.detach().cpu().item()),
            "action_mse": float(action_mse.detach().cpu().item()),
            "masked_tokens": float(masked_tokens.detach().cpu().item()),
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
