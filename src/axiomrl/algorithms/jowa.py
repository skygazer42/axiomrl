from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.dqn import DQN, _weighted_smooth_l1_loss
from axiomrl.models.cnn.jowa_q_network import CNNJOWAQNetwork


class JOWA(DQN):
    def __init__(
        self,
        *,
        q_network: CNNJOWAQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        jowa_world_model_loss_coef: float = 1.0,
        jowa_reward_loss_coef: float = 1.0,
        jowa_reconstruction_loss_coef: float = 1.0,
        jowa_consistency_loss_coef: float = 0.5,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )
        self.jowa_world_model_loss_coef = float(jowa_world_model_loss_coef)
        self.jowa_reward_loss_coef = float(jowa_reward_loss_coef)
        self.jowa_reconstruction_loss_coef = float(jowa_reconstruction_loss_coef)
        self.jowa_consistency_loss_coef = float(jowa_consistency_loss_coef)

    @property
    def jowa_network(self) -> CNNJOWAQNetwork:
        return self.q_network

    @property
    def target_jowa_network(self) -> CNNJOWAQNetwork:
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

        online_latent = self.jowa_network.encode(obs)
        predicted_next_latent = self.jowa_network.transition(online_latent, actions)
        predicted_rewards = self.jowa_network.predict_reward(predicted_next_latent)
        reconstructed_next_obs = self.jowa_network.decode(predicted_next_latent)

        reward_loss = F.mse_loss(predicted_rewards, rewards)
        reconstruction_target = next_obs / 255.0
        reconstruction_loss = F.mse_loss(reconstructed_next_obs, reconstruction_target)

        with torch.no_grad():
            target_next_latent = self.target_jowa_network.encode(next_obs)
        consistency_loss = F.mse_loss(predicted_next_latent, target_next_latent)

        model_loss = (
            self.jowa_reward_loss_coef * reward_loss
            + self.jowa_reconstruction_loss_coef * reconstruction_loss
            + self.jowa_consistency_loss_coef * consistency_loss
        )
        loss = q_loss + self.jowa_world_model_loss_coef * model_loss

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
            "jowa_model_loss": float(model_loss.detach().cpu().item()),
            "jowa_reward_loss": float(reward_loss.detach().cpu().item()),
            "jowa_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "jowa_consistency_loss": float(consistency_loss.detach().cpu().item()),
            "jowa_world_model_loss_coef": self.jowa_world_model_loss_coef,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["jowa_world_model_loss_coef"] = self.jowa_world_model_loss_coef
        state["jowa_reward_loss_coef"] = self.jowa_reward_loss_coef
        state["jowa_reconstruction_loss_coef"] = self.jowa_reconstruction_loss_coef
        state["jowa_consistency_loss_coef"] = self.jowa_consistency_loss_coef
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.jowa_world_model_loss_coef = float(
            state_dict.get("jowa_world_model_loss_coef", self.jowa_world_model_loss_coef)
        )
        self.jowa_reward_loss_coef = float(state_dict.get("jowa_reward_loss_coef", self.jowa_reward_loss_coef))
        self.jowa_reconstruction_loss_coef = float(
            state_dict.get("jowa_reconstruction_loss_coef", self.jowa_reconstruction_loss_coef)
        )
        self.jowa_consistency_loss_coef = float(
            state_dict.get("jowa_consistency_loss_coef", self.jowa_consistency_loss_coef)
        )
