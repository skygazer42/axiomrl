from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.dreamer import Dreamer
from rl_training.models.po_dreamer import PODreamerModel


class PODreamer(Dreamer):
    def __init__(
        self,
        *,
        model: PODreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        memory_loss_coef: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        if float(memory_loss_coef) < 0.0:
            raise ValueError(f"memory_loss_coef must be >= 0, got {memory_loss_coef}")
        self.memory_loss_coef = float(memory_loss_coef)

    @property
    def po_dreamer_model(self) -> PODreamerModel:
        return self.model

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        features = self.po_dreamer_model.encode(obs)
        memory_features = self.po_dreamer_model.encode_memory(obs)
        fused_features, memory_gate = self.po_dreamer_model.fuse_with_memory(
            features,
            memory_features,
            detach_memory=False,
        )

        next_features = self.po_dreamer_model.dynamics_step(fused_features, actions)
        predicted_next_memory = self.po_dreamer_model.predict_memory(next_features)
        fused_next_features, next_memory_gate = self.po_dreamer_model.fuse_with_memory(
            next_features,
            predicted_next_memory,
            detach_memory=False,
        )

        decoded_next = self.po_dreamer_model.decode(fused_next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.po_dreamer_model.predict_reward(fused_next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        with torch.no_grad():
            target_next_memory = self.po_dreamer_model.encode_memory(next_obs)
        memory_loss = F.mse_loss(predicted_next_memory, target_next_memory)

        loss = reconstruction_loss + reward_loss + self.memory_loss_coef * memory_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        gate_mean = 0.5 * (memory_gate.mean() + next_memory_gate.mean())
        metrics = {
            "po_dreamer_world_model_loss": float(loss.detach().cpu().item()),
            "po_dreamer_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "po_dreamer_reward_loss": float(reward_loss.detach().cpu().item()),
            "po_dreamer_memory_loss": float(memory_loss.detach().cpu().item()),
            "po_dreamer_memory_gate_mean": float(gate_mean.detach().cpu().item()),
            "po_dreamer_memory_loss_coef": self.memory_loss_coef,
            "po_dreamer_memory_mix": self.po_dreamer_model.memory_mix,
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
            base_features = self.po_dreamer_model.encode(start_obs.to(dtype=torch.float32))
            memory_state = self.po_dreamer_model.encode_memory(start_obs.to(dtype=torch.float32))
            features, initial_gate = self.po_dreamer_model.fuse_with_memory(
                base_features,
                memory_state,
                detach_memory=True,
            )

        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []
        gate_terms: list[torch.Tensor] = [initial_gate.mean()]

        for _ in range(horizon):
            logits = self.po_dreamer_model.actor_logits(features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())
            value_terms.append(self.po_dreamer_model.value(features))

            with torch.no_grad():
                next_features = self.po_dreamer_model.dynamics_step(features, actions)
                predicted_memory = self.po_dreamer_model.predict_memory(next_features)
                memory_state = torch.lerp(
                    memory_state,
                    predicted_memory,
                    self.po_dreamer_model.memory_mix,
                )
                features, memory_gate = self.po_dreamer_model.fuse_with_memory(
                    next_features,
                    memory_state,
                    detach_memory=True,
                )
                reward_terms.append(self.po_dreamer_model.predict_reward(features))
                gate_terms.append(memory_gate.mean())

        with torch.no_grad():
            bootstrap_value = self.po_dreamer_model.value(features).detach()

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
            "po_dreamer_actor_loss": float(actor_loss.detach().cpu().item()),
            "po_dreamer_critic_loss": float(critic_loss.detach().cpu().item()),
            "po_dreamer_imagination_horizon": float(horizon),
            "po_dreamer_imagination_reward_mean": float(
                torch.stack(reward_terms, dim=0).mean().detach().cpu().item()
            )
            if reward_terms
            else 0.0,
            "po_dreamer_memory_gate_mean": float(torch.stack(gate_terms).mean().detach().cpu().item()),
            "po_dreamer_memory_mix": self.po_dreamer_model.memory_mix,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["memory_loss_coef"] = self.memory_loss_coef
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.memory_loss_coef = float(state_dict.get("memory_loss_coef", self.memory_loss_coef))
