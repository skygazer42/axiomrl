from __future__ import annotations

from typing import Any

import numpy as np
import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_mopo import MLPMOPOEnsembleModel


def _pets_model_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = batch["targets"].unsqueeze(0).expand_as(batch["predicted_means"])
    inv_vars = torch.exp(-batch["predicted_logvars"])
    nll = 0.5 * ((targets - batch["predicted_means"]).pow(2) * inv_vars + batch["predicted_logvars"])
    reward_mae = (batch["predicted_means"][..., -1] - targets[..., -1]).abs().mean()
    delta_obs_mae = (batch["predicted_means"][..., :-1] - targets[..., :-1]).abs().mean()
    return {
        "pets_model_loss": nll.mean(),
        "reward_mae": reward_mae,
        "delta_obs_mae": delta_obs_mae,
        "ensemble_disagreement": batch["ensemble_disagreement"].mean(),
    }


def pets_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "predicted_means": torch.as_tensor(batch["predicted_means"], dtype=torch.float32),
        "predicted_logvars": torch.as_tensor(batch["predicted_logvars"], dtype=torch.float32),
        "targets": torch.as_tensor(batch["targets"], dtype=torch.float32),
        "ensemble_disagreement": torch.as_tensor(batch["ensemble_disagreement"], dtype=torch.float32),
    }
    terms = _pets_model_loss_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class PETS:
    def __init__(
        self,
        *,
        dynamics_model: MLPMOPOEnsembleModel,
        learning_rate: float,
    ) -> None:
        if float(learning_rate) <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

        self.model = dynamics_model
        self.dynamics_model = dynamics_model
        self.policy = dynamics_model
        self.optimizer = torch.optim.Adam(
            self.dynamics_model.ensemble_parameters(), lr=float(learning_rate), weight_decay=0.0
        )

    def _device(self) -> torch.device:
        parameter = next(iter(self.dynamics_model.parameters()), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        device = self._device()
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)

        predicted_means, predicted_logvars = self.dynamics_model.predict_distribution(obs, actions)
        targets = torch.cat([next_obs - obs, rewards.unsqueeze(-1)], dim=-1)
        disagreement = predicted_means.std(dim=0).mean(dim=-1)
        terms = _pets_model_loss_terms(
            {
                "predicted_means": predicted_means,
                "predicted_logvars": predicted_logvars,
                "targets": targets,
                "ensemble_disagreement": disagreement,
            }
        )

        self.optimizer.zero_grad(set_to_none=True)
        terms["pets_model_loss"].backward()
        self.optimizer.step()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def _evaluate_action_sequences(
        self,
        obs: torch.Tensor,
        action_sequences: torch.Tensor,
        *,
        num_particles: int,
        deterministic: bool,
    ) -> torch.Tensor:
        num_candidates, horizon, _ = action_sequences.shape
        obs_particles = obs.unsqueeze(0).repeat(num_candidates * num_particles, 1)
        total_rewards = torch.zeros(num_candidates * num_particles, dtype=torch.float32, device=action_sequences.device)

        for step in range(horizon):
            action_step = action_sequences[:, step, :].repeat_interleave(num_particles, dim=0)
            predicted_means, predicted_logvars = self.dynamics_model.predict_distribution(obs_particles, action_step)
            if deterministic:
                transitions = predicted_means.mean(dim=0)
            else:
                std = torch.exp(0.5 * predicted_logvars)
                batch_indices = torch.arange(obs_particles.shape[0], device=obs_particles.device)
                ensemble_indices = torch.randint(
                    0,
                    predicted_means.shape[0],
                    (obs_particles.shape[0],),
                    device=obs_particles.device,
                )
                chosen_means = predicted_means[ensemble_indices, batch_indices]
                chosen_std = std[ensemble_indices, batch_indices]
                transitions = chosen_means + torch.randn_like(chosen_std) * chosen_std

            delta_obs = transitions[:, : obs_particles.shape[1]]
            rewards = transitions[:, obs_particles.shape[1]]
            obs_particles = obs_particles + delta_obs
            total_rewards = total_rewards + rewards

        return total_rewards.view(num_candidates, num_particles).mean(dim=1)

    def plan_action(
        self,
        obs: object,
        *,
        action_low: object,
        action_high: object,
        horizon: int,
        num_candidates: int,
        num_iterations: int,
        num_topk: int,
        num_particles: int = 8,
        deterministic: bool = True,
    ) -> np.ndarray:
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if int(num_candidates) < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if int(num_iterations) < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if int(num_topk) < 1:
            raise ValueError(f"num_topk must be >= 1, got {num_topk}")
        if int(num_particles) < 1:
            raise ValueError(f"num_particles must be >= 1, got {num_particles}")
        if int(num_topk) > int(num_candidates):
            raise ValueError("num_topk must be <= num_candidates")

        device = self._device()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 2:
            if int(obs_tensor.shape[0]) != 1:
                raise ValueError(
                    f"plan_action currently supports a single observation, got shape {tuple(obs_tensor.shape)!r}"
                )
            obs_tensor = obs_tensor[0]
        elif obs_tensor.ndim != 1:
            raise ValueError(f"expected flat observation for planning, got shape {tuple(obs_tensor.shape)!r}")

        low = torch.as_tensor(action_low, dtype=torch.float32, device=device).reshape(1, 1, -1)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=device).reshape(1, 1, -1)
        action_dim = int(low.shape[-1])
        candidate_low = low.expand(int(num_candidates), int(horizon), action_dim)
        candidate_high = high.expand(int(num_candidates), int(horizon), action_dim)

        mean = 0.5 * (low + high)
        mean = mean.expand(1, int(horizon), action_dim).clone()
        std = 0.5 * (high - low)
        std = std.expand_as(mean).clone()
        min_std = torch.clamp(0.05 * (high - low), min=1e-3)

        self.set_eval_mode()
        with torch.no_grad():
            for _ in range(int(num_iterations)):
                candidate_sequences = mean + std * torch.randn(
                    (int(num_candidates), int(horizon), action_dim),
                    dtype=torch.float32,
                    device=device,
                )
                candidate_sequences = torch.max(torch.min(candidate_sequences, candidate_high), candidate_low)
                values = self._evaluate_action_sequences(
                    obs_tensor,
                    candidate_sequences,
                    num_particles=int(num_particles),
                    deterministic=deterministic,
                )
                elite_indices = torch.topk(values, k=int(num_topk)).indices
                elite_sequences = candidate_sequences[elite_indices]
                mean = elite_sequences.mean(dim=0, keepdim=True)
                std = torch.maximum(elite_sequences.std(dim=0, unbiased=False, keepdim=True), min_std)

        action = mean[0, 0]
        action = torch.max(torch.min(action, high[0, 0]), low[0, 0])
        return action.detach().cpu().numpy().astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {
            "dynamics_model": self.dynamics_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.dynamics_model.load_state_dict(state_dict["dynamics_model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.dynamics_model.train(True)

    def set_eval_mode(self) -> None:
        self.dynamics_model.eval()
