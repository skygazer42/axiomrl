from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.dreamer import DreamerModel
from axiomrl.policies.base import PolicyOutput


class Dreamer:
    def __init__(
        self,
        *,
        model: DreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
    ) -> None:
        self.model = model
        self.policy = model
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)

        self.world_model_optimizer = torch.optim.Adam(
            self.model.parameters_world_model(),
            lr=float(world_model_learning_rate),
            weight_decay=0.0,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.parameters_actor(),
            lr=float(actor_learning_rate),
            weight_decay=0.0,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.parameters_critic(),
            lr=float(critic_learning_rate),
            weight_decay=0.0,
        )

    def update_world_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)

        obs_target = next_obs / 255.0
        features = self.model.encode(obs)
        next_features = self.model.dynamics_step(features, actions)
        decoded_next = self.model.decode(next_features)
        reconstruction_loss = F.mse_loss(decoded_next, obs_target)

        predicted_rewards = self.model.predict_reward(next_features)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        loss = reconstruction_loss + reward_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()

        metrics = {
            "dreamer_world_model_loss": float(loss.detach().cpu().item()),
            "dreamer_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "dreamer_reward_loss": float(reward_loss.detach().cpu().item()),
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
            start_features = self.model.encode(start_obs.to(dtype=torch.float32))

        features = start_features.detach()
        logprob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        reward_terms: list[torch.Tensor] = []

        for _ in range(horizon):
            logits = self.model.actor_logits(features)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()

            logprob_terms.append(distribution.log_prob(actions))
            entropy_terms.append(distribution.entropy())
            value_terms.append(self.model.value(features))

            with torch.no_grad():
                next_features = self.model.dynamics_step(features, actions)
                reward_terms.append(self.model.predict_reward(next_features))
                features = next_features.detach()

        with torch.no_grad():
            bootstrap_value = self.model.value(features).detach()

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
            "dreamer_actor_loss": float(actor_loss.detach().cpu().item()),
            "dreamer_critic_loss": float(critic_loss.detach().cpu().item()),
            "dreamer_imagination_horizon": float(horizon),
            "dreamer_imagination_reward_mean": float(torch.stack(reward_terms, dim=0).mean().detach().cpu().item())
            if reward_terms
            else 0.0,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        return self.update_world_model(batch, global_step=global_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.world_model_optimizer.load_state_dict(state_dict["world_model_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.gamma = float(state_dict.get("gamma", self.gamma))
        self.entropy_coef = float(state_dict.get("entropy_coef", self.entropy_coef))

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self.model.act(obs, state=state, deterministic=deterministic)

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
