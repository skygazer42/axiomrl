import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.recurrent import LSTMQNetwork


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    normalizer = mask.sum().clamp_min(1.0)
    return masked.sum() / normalizer


def _drqn_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    chosen_q_values = torch.as_tensor(batch["chosen_q_values"], dtype=torch.float32)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=chosen_q_values.device)
    mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=chosen_q_values.device)
    td_error = target_q_values - chosen_q_values
    per_item_loss = F.smooth_l1_loss(chosen_q_values, target_q_values, reduction="none")
    loss = _masked_mean(per_item_loss, mask)

    return {
        "loss": loss,
        "q_value_mean": _masked_mean(chosen_q_values, mask),
        "target_mean": _masked_mean(target_q_values, mask),
        "td_error_mean": _masked_mean(td_error.abs(), mask),
    }


def drqn_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _drqn_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class DRQN:
    def __init__(
        self,
        *,
        q_network: LSTMQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        double_q: bool = False,
    ) -> None:
        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.double_q = bool(double_q)

    def _extended_sequence_inputs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=obs.device)
        episode_starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32, device=obs.device)
        extended_obs = torch.cat([obs, next_obs[-1:].clone()], dim=0)
        final_episode_start = dones[-1:].clone()
        extended_episode_starts = torch.cat([episode_starts, final_episode_start], dim=0)
        return extended_obs, extended_episode_starts

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

        terms = _drqn_loss_terms(
            {
                "chosen_q_values": chosen_q_values,
                "target_q_values": target_q_values,
                "mask": mask,
            }
        )

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
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
