from __future__ import annotations

import copy
from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from rl_training.models.mlp_dueling_q_network import MLPDuelingQNetwork
from rl_training.models.mlp_noisy_q_network import MLPNoisyQNetwork
from rl_training.models.mlp_q_network import MLPQNetwork


def _dqn_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    chosen_q_values = batch["q_values"].gather(1, batch["actions"].long().unsqueeze(-1)).squeeze(-1)
    td_error = batch["target_q_values"] - chosen_q_values
    loss = F.smooth_l1_loss(chosen_q_values, batch["target_q_values"])

    return {
        "loss": loss,
        "q_value_mean": chosen_q_values.mean(),
        "target_mean": batch["target_q_values"].mean(),
        "td_error_mean": td_error.abs().mean(),
    }


def dqn_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _dqn_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class DQN:
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        double_q: bool = False,
    ) -> None:
        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.double_q = double_q
        self.last_td_errors: torch.Tensor | None = None

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_network(obs)
        with torch.no_grad():
            if self.double_q:
                next_actions = self.q_network(next_obs).argmax(dim=-1, keepdim=True)
                next_q_values = self.target_network(next_obs).gather(1, next_actions).squeeze(-1)
            else:
                next_q_values = self.target_network(next_obs).max(dim=-1).values
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        td_errors = target_q_values - chosen_q_values
        self.last_td_errors = td_errors.detach().abs()

        per_item_losses = F.smooth_l1_loss(chosen_q_values, target_q_values, reduction="none")
        weights = batch.get("weights")
        if weights is None:
            loss = per_item_losses.mean()
        else:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=per_item_losses.device)
            if weight_tensor.ndim != 1:
                weight_tensor = weight_tensor.reshape(-1)
            loss = (weight_tensor * per_item_losses).mean()

        terms = {
            "loss": loss,
            "q_value_mean": chosen_q_values.mean(),
            "target_mean": target_q_values.mean(),
            "td_error_mean": td_errors.abs().mean(),
        }

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


class DoubleDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )


class RainbowDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )


class DuelingDQN(DQN):
    pass


class NoisyDQN(DQN):
    pass


class PrioritizedDQN(DQN):
    pass
