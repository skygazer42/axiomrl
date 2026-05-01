from typing import Any

import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_ars import MLPARSModel


def _centered_ranks(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"expected 1D values for centered ranks, got {tuple(values.shape)!r}")
    if values.numel() <= 1:
        return torch.zeros_like(values)

    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    ranks /= float(values.numel() - 1)
    return ranks - 0.5


def _openai_es_metric_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "reward_std": batch["reward_std"],
        "utility_mean": batch["utilities"].mean(),
        "update_norm": batch["update"].norm(),
        "parameter_norm": batch["parameters"].norm(),
        "positive_return_mean": batch["positive_returns"].mean(),
        "negative_return_mean": batch["negative_returns"].mean(),
    }


def openai_es_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "positive_returns": torch.as_tensor(batch["positive_returns"], dtype=torch.float32),
        "negative_returns": torch.as_tensor(batch["negative_returns"], dtype=torch.float32),
        "utilities": torch.as_tensor(batch["utilities"], dtype=torch.float32),
        "reward_std": torch.as_tensor(batch["reward_std"], dtype=torch.float32),
        "update": torch.as_tensor(batch["update"], dtype=torch.float32),
        "parameters": torch.as_tensor(batch["parameters"], dtype=torch.float32),
    }
    terms = _openai_es_metric_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class OpenAIES:
    def __init__(
        self,
        *,
        model: MLPARSModel,
        step_size: float,
        noise_std: float,
        weight_decay: float = 0.0,
    ) -> None:
        if float(step_size) <= 0.0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        if float(noise_std) <= 0.0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")
        if float(weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        self.model = model
        self.policy = model
        self.step_size = float(step_size)
        self.noise_std = float(noise_std)
        self.weight_decay = float(weight_decay)

    def sample_perturbations(self, num_directions: int) -> torch.Tensor:
        if int(num_directions) <= 0:
            raise ValueError(f"num_directions must be > 0, got {num_directions}")
        parameters = self.model.flat_parameters()
        return torch.randn(
            (int(num_directions), int(parameters.numel())),
            dtype=parameters.dtype,
            device=parameters.device,
        )

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        parameters = self.model.flat_parameters()
        perturbations = torch.as_tensor(batch["perturbations"], dtype=torch.float32, device=parameters.device)
        positive_returns = torch.as_tensor(batch["positive_returns"], dtype=torch.float32, device=parameters.device)
        negative_returns = torch.as_tensor(batch["negative_returns"], dtype=torch.float32, device=parameters.device)

        if perturbations.ndim != 2:
            raise ValueError(f"expected perturbations to have shape [K, P], got {tuple(perturbations.shape)!r}")
        if positive_returns.shape != negative_returns.shape:
            raise ValueError("positive_returns and negative_returns must have the same shape")
        if positive_returns.ndim != 1:
            raise ValueError(f"expected returns to have shape [K], got {tuple(positive_returns.shape)!r}")
        if perturbations.shape[0] != positive_returns.shape[0]:
            raise ValueError("perturbations and returns must agree on direction count")

        all_returns = torch.cat([positive_returns, negative_returns], dim=0)
        ranked_utilities = _centered_ranks(all_returns)
        num_directions = int(perturbations.shape[0])
        positive_utilities = ranked_utilities[:num_directions]
        negative_utilities = ranked_utilities[num_directions:]
        utilities = positive_utilities - negative_utilities
        search_direction = torch.matmul(utilities, perturbations) / float(num_directions)
        update = (self.step_size / self.noise_std) * search_direction
        if self.weight_decay > 0.0:
            update = update - self.weight_decay * parameters

        updated_parameters = parameters + update.to(device=parameters.device, dtype=parameters.dtype)
        self.model.set_flat_parameters(updated_parameters)

        reward_std = torch.std(all_returns, dim=0, unbiased=False)
        terms = _openai_es_metric_terms(
            {
                "positive_returns": positive_returns,
                "negative_returns": negative_returns,
                "utilities": utilities,
                "reward_std": reward_std,
                "update": update,
                "parameters": updated_parameters,
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
