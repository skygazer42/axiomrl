from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.dreamer import Dreamer
from rl_training.models.mow import MoWModel


class MoW(Dreamer):
    def __init__(
        self,
        *,
        model: MoWModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )

    @property
    def mow_model(self) -> MoWModel:
        return self.model

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        features = self.mow_model.encode(obs)
        next_features, dynamics_gate_probs = self.mow_model.dynamics_step_with_gates(features, actions)
        decoded_next = self.mow_model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards, reward_gate_probs = self.mow_model.predict_reward_with_gates(next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        loss = reconstruction_loss + reward_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        gate_entropy = 0.5 * (
            self.mow_model.gate_entropy(dynamics_gate_probs).mean()
            + self.mow_model.gate_entropy(reward_gate_probs).mean()
        )
        metrics = {
            "mow_world_model_loss": float(loss.detach().cpu().item()),
            "mow_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "mow_reward_loss": float(reward_loss.detach().cpu().item()),
            "mow_gate_entropy": float(gate_entropy.detach().cpu().item()),
            "mow_dynamics_gate_entropy": float(self.mow_model.gate_entropy(dynamics_gate_probs).mean().detach().cpu().item()),
            "mow_reward_gate_entropy": float(self.mow_model.gate_entropy(reward_gate_probs).mean().detach().cpu().item()),
            "mow_num_experts": float(self.mow_model.num_experts),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def update_actor_critic(
        self,
        start_obs: torch.Tensor,
        *,
        imagination_horizon: int,
        global_step: int,
    ) -> UpdateResult:
        del global_step
        self.set_train_mode()

        horizon = max(1, int(imagination_horizon))
        with torch.no_grad():
            start_features = self.mow_model.encode(start_obs.to(dtype=torch.float32))

        features = start_features.detach()
        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []
        gate_entropies: list[torch.Tensor] = []

        for _ in range(horizon):
            logits, actor_gate_probs = self.mow_model.actor_logits_with_gates(features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())

            values, critic_gate_probs = self.mow_model.value_with_gates(features)
            value_terms.append(values)
            gate_entropies.append(0.5 * (self.mow_model.gate_entropy(actor_gate_probs) + self.mow_model.gate_entropy(critic_gate_probs)))

            with torch.no_grad():
                next_features, dynamics_gate_probs = self.mow_model.dynamics_step_with_gates(features, actions)
                predicted_rewards, reward_gate_probs = self.mow_model.predict_reward_with_gates(next_features)
                reward_terms.append(predicted_rewards)
                gate_entropies.append(
                    0.5 * (self.mow_model.gate_entropy(dynamics_gate_probs) + self.mow_model.gate_entropy(reward_gate_probs))
                )
                features = next_features.detach()

        with torch.no_grad():
            bootstrap_value, bootstrap_gate_probs = self.mow_model.value_with_gates(features)
            bootstrap_value = bootstrap_value.detach()
            gate_entropies.append(self.mow_model.gate_entropy(bootstrap_gate_probs))

        returns: list[torch.Tensor] = []
        running_return = bootstrap_value
        for reward in reversed(reward_terms):
            running_return = reward + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        returns_tensor = torch.stack(returns, dim=0)
        values_tensor = torch.stack(value_terms, dim=0)
        advantages = returns_tensor - values_tensor

        logprobs_tensor = torch.stack(logprob_terms, dim=0)
        entropy_tensor = torch.stack(entropy_terms, dim=0)

        actor_loss = -(logprobs_tensor * advantages.detach()).mean() - self.entropy_coef * entropy_tensor.mean()
        critic_loss = 0.5 * F.mse_loss(values_tensor, returns_tensor.detach())

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        metrics = {
            "mow_actor_loss": float(actor_loss.detach().cpu().item()),
            "mow_critic_loss": float(critic_loss.detach().cpu().item()),
            "mow_imagination_horizon": float(horizon),
            "mow_imagination_reward_mean": float(torch.stack(reward_terms, dim=0).mean().detach().cpu().item()) if reward_terms else 0.0,
            "mow_gate_entropy": float(torch.stack(gate_entropies, dim=0).mean().detach().cpu().item()) if gate_entropies else 0.0,
            "mow_num_experts": float(self.mow_model.num_experts),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["num_experts"] = self.mow_model.num_experts
        state["gating_hidden_size"] = self.mow_model.gating_hidden_size
        return state
