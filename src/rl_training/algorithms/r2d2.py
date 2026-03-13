from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.drqn import DRQN
from rl_training.models.recurrent import LSTMQNetwork


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    normalizer = mask.sum().clamp_min(1.0)
    return masked.sum() / normalizer


def _per_sequence_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    normalizer = mask.sum(dim=0).clamp_min(1.0)
    return masked.sum(dim=0) / normalizer


def _r2d2_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    chosen_q_values = torch.as_tensor(batch["chosen_q_values"], dtype=torch.float32)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=chosen_q_values.device)
    mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=chosen_q_values.device)
    td_error = target_q_values - chosen_q_values
    per_item_loss = F.smooth_l1_loss(chosen_q_values, target_q_values, reduction="none")
    per_sequence_loss = _per_sequence_masked_mean(per_item_loss, mask)

    weights = batch.get("weights")
    if weights is None:
        loss = per_sequence_loss.mean()
    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=chosen_q_values.device).reshape(-1)
        loss = (weight_tensor * per_sequence_loss).mean()

    return {
        "loss": loss,
        "q_value_mean": _masked_mean(chosen_q_values, mask),
        "target_mean": _masked_mean(target_q_values, mask),
        "td_error_mean": _masked_mean(td_error.abs(), mask),
    }


def r2d2_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _r2d2_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class R2D2(DRQN):
    def __init__(
        self,
        *,
        q_network: LSTMQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        double_q: bool = True,
        priority_eta: float = 0.9,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=double_q,
        )
        self.priority_eta = float(priority_eta)
        self.last_sequence_priorities: torch.Tensor | None = None

    def _sequence_priorities(self, td_errors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        abs_errors = td_errors.abs() * mask
        max_errors = abs_errors.max(dim=0).values
        mean_errors = _per_sequence_masked_mean(abs_errors, mask)
        return self.priority_eta * max_errors + (1.0 - self.priority_eta) * mean_errors

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()
        device = next(self.q_network.parameters()).device

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
        episode_starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)
        initial_state = (
            torch.as_tensor(batch["initial_h"], dtype=torch.float32, device=device),
            torch.as_tensor(batch["initial_c"], dtype=torch.float32, device=device),
        )

        q_values = self.q_network.q_values_sequence(
            obs,
            initial_state=initial_state,
            episode_starts=episode_starts,
        )
        chosen_q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        extended_obs, extended_episode_starts = self._extended_sequence_inputs(
            {
                "obs": obs,
                "next_obs": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device),
                "dones": dones,
                "episode_starts": episode_starts,
            }
        )

        with torch.no_grad():
            extended_target_q_values = self.target_network.q_values_sequence(
                extended_obs,
                initial_state=initial_state,
                episode_starts=extended_episode_starts,
            )
            if self.double_q:
                extended_online_q_values = self.q_network.q_values_sequence(
                    extended_obs,
                    initial_state=initial_state,
                    episode_starts=extended_episode_starts,
                )
                next_actions = extended_online_q_values[1:].argmax(dim=-1, keepdim=True)
                next_target_q_values = extended_target_q_values[1:].gather(-1, next_actions).squeeze(-1)
            else:
                next_target_q_values = extended_target_q_values[1:].max(dim=-1).values
            target_q_values = rewards + self.gamma * next_target_q_values * (1.0 - dones)

        td_errors = target_q_values - chosen_q_values
        self.last_sequence_priorities = self._sequence_priorities(td_errors.detach(), mask.detach())

        terms = _r2d2_loss_terms(
            {
                "chosen_q_values": chosen_q_values,
                "target_q_values": target_q_values,
                "mask": mask,
                "weights": batch.get("weights"),
            }
        )

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)
