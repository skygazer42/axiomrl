from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.dqn import DQN, _weighted_smooth_l1_loss
from rl_training.models.cnn.spr_q_network import CNNSPRQNetwork


class SPR(DQN):
    def __init__(
        self,
        *,
        q_network: CNNSPRQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        spr_loss_coef: float = 1.0,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )
        self.spr_loss_coef = float(spr_loss_coef)

    @property
    def spr_network(self) -> CNNSPRQNetwork:
        return self.q_network

    @property
    def target_spr_network(self) -> CNNSPRQNetwork:
        return self.target_network

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()
        device = next(self.q_network.parameters()).device

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

        q_values = self.q_network(obs)
        with torch.no_grad():
            target_q_values = self._compute_target_q_values(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                q_values=q_values,
            )

        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        td_errors = target_q_values - chosen_q_values
        self.last_td_errors = td_errors.detach().abs()
        q_loss = _weighted_smooth_l1_loss(chosen_q_values, target_q_values, weights=batch.get("weights"))

        online_latent = self.spr_network.encode(obs)
        predicted_next_latent = self.spr_network.transition(online_latent, actions)
        online_projection = F.normalize(self.spr_network.predict_projection(predicted_next_latent), dim=-1)

        with torch.no_grad():
            target_latent = self.target_spr_network.encode(next_obs)
            target_projection = F.normalize(self.target_spr_network.project(target_latent), dim=-1)

        transition_mask = 1.0 - dones
        cosine_similarity = (online_projection * target_projection).sum(dim=-1)
        spr_error = 1.0 - cosine_similarity
        spr_loss = (spr_error * transition_mask).sum() / transition_mask.sum().clamp_min(1.0)

        loss = q_loss + self.spr_loss_coef * spr_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "q_loss": float(q_loss.detach().cpu().item()),
            "q_value_mean": float(chosen_q_values.mean().detach().cpu().item()),
            "target_mean": float(target_q_values.mean().detach().cpu().item()),
            "td_error_mean": float(td_errors.abs().mean().detach().cpu().item()),
            "spr_loss": float(spr_loss.detach().cpu().item()),
            "spr_cosine_similarity": float(cosine_similarity.mean().detach().cpu().item()),
            "spr_loss_coef": self.spr_loss_coef,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["spr_loss_coef"] = self.spr_loss_coef
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.spr_loss_coef = float(state_dict.get("spr_loss_coef", self.spr_loss_coef))
