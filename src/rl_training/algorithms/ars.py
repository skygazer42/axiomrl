from __future__ import annotations

from typing import Any

import torch

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_ars import MLPARSModel


def _ars_metric_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "reward_std": batch["reward_std"],
        "update_norm": batch["update"].norm(),
        "parameter_norm": batch["parameters"].norm(),
        "positive_return_mean": batch["positive_returns"].mean(),
        "negative_return_mean": batch["negative_returns"].mean(),
        "selected_directions": batch["selected_directions"],
    }


def ars_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "positive_returns": torch.as_tensor(batch["positive_returns"], dtype=torch.float32),
        "negative_returns": torch.as_tensor(batch["negative_returns"], dtype=torch.float32),
        "reward_std": torch.as_tensor(batch["reward_std"], dtype=torch.float32),
        "update": torch.as_tensor(batch["update"], dtype=torch.float32),
        "parameters": torch.as_tensor(batch["parameters"], dtype=torch.float32),
        "selected_directions": torch.as_tensor(batch["selected_directions"], dtype=torch.float32),
    }
    terms = _ars_metric_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class ARS:
    def __init__(
        self,
        *,
        model: MLPARSModel,
        step_size: float,
        noise_std: float,
        num_top_directions: int,
    ) -> None:
        if float(step_size) <= 0.0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        if float(noise_std) <= 0.0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")
        if int(num_top_directions) <= 0:
            raise ValueError(f"num_top_directions must be > 0, got {num_top_directions}")

        self.model = model
        self.policy = model
        self.step_size = float(step_size)
        self.noise_std = float(noise_std)
        self.num_top_directions = int(num_top_directions)

    def sample_perturbations(self, num_directions: int) -> torch.Tensor:
        if int(num_directions) <= 0:
            raise ValueError(f"num_directions must be > 0, got {num_directions}")
        return torch.randn(
            (int(num_directions), self.model.num_parameters),
            dtype=self.model.flat_parameters().dtype,
            device=self.model.flat_parameters().device,
        )

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        perturbations = torch.as_tensor(batch["perturbations"], dtype=torch.float32, device=self.model.flat_parameters().device)
        positive_returns = torch.as_tensor(batch["positive_returns"], dtype=torch.float32, device=perturbations.device)
        negative_returns = torch.as_tensor(batch["negative_returns"], dtype=torch.float32, device=perturbations.device)

        if perturbations.ndim != 2:
            raise ValueError(f"expected perturbations to have shape [K, P], got {tuple(perturbations.shape)!r}")
        if positive_returns.shape != negative_returns.shape:
            raise ValueError("positive_returns and negative_returns must have the same shape")
        if positive_returns.ndim != 1:
            raise ValueError(f"expected returns to have shape [K], got {tuple(positive_returns.shape)!r}")
        if perturbations.shape[0] != positive_returns.shape[0]:
            raise ValueError("perturbations and returns must agree on direction count")

        selected_count = min(self.num_top_directions, int(perturbations.shape[0]))
        scores = torch.maximum(positive_returns, negative_returns)
        selected_indices = torch.topk(scores, k=selected_count).indices
        selected_perturbations = perturbations[selected_indices]
        selected_positive_returns = positive_returns[selected_indices]
        selected_negative_returns = negative_returns[selected_indices]
        reward_std = torch.std(
            torch.cat([selected_positive_returns, selected_negative_returns], dim=0),
            correction=0,
        )
        reward_scale = torch.clamp(reward_std, min=1e-8)
        search_direction = ((selected_positive_returns - selected_negative_returns).unsqueeze(-1) * selected_perturbations).mean(dim=0)
        update = (self.step_size / reward_scale) * search_direction

        parameters = self.model.flat_parameters()
        updated_parameters = parameters + update.to(device=parameters.device, dtype=parameters.dtype)
        self.model.set_flat_parameters(updated_parameters)

        terms = _ars_metric_terms(
            {
                "positive_returns": selected_positive_returns,
                "negative_returns": selected_negative_returns,
                "reward_std": reward_std,
                "update": update,
                "parameters": updated_parameters,
                "selected_directions": torch.tensor(float(selected_count), dtype=torch.float32, device=update.device),
            }
        )
        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
