import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.c51_dqn import _project_distribution
from axiomrl.models.mlp_d4pg import MLPD4PGModel


def _d4pg_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    log_probs = F.log_softmax(batch["logits"], dim=-1)
    critic_loss = -(batch["target_distributions"] * log_probs).sum(dim=-1).mean()
    actor_loss = -batch["actor_q_values"].mean()
    q_values = batch.get("q_values", batch["actor_q_values"])
    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "q_value_mean": q_values.mean(),
    }


def d4pg_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _d4pg_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class D4PG:
    def __init__(
        self,
        *,
        model: MLPD4PGModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        v_min: float,
        v_max: float,
        num_atoms: int,
    ) -> None:
        if num_atoms != model.num_atoms:
            raise ValueError(f"expected num_atoms={model.num_atoms}, got {num_atoms}")
        if abs(float(v_min) - float(model.v_min)) > 1e-8 or abs(float(v_max) - float(model.v_max)) > 1e-8:
            raise ValueError("D4PG v_min/v_max must match the model support range")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=learning_rate, weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.tau = tau
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_atoms = int(num_atoms)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        logits = self.model.distribution_logits(obs, actions)
        q_values = (F.softmax(logits, dim=-1) * self.model.support).sum(dim=-1)

        with torch.no_grad():
            next_actions = self.target_model.actor(next_obs)
            next_action_probs = self.target_model.probabilities(next_obs, next_actions)
            target_distributions = _project_distribution(
                next_action_probs=next_action_probs,
                rewards=rewards,
                dones=dones,
                support=self.model.support,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
            )
            target_q_values = (target_distributions * self.model.support).sum(dim=-1)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _d4pg_loss_terms(
            {
                "logits": logits,
                "target_distributions": target_distributions,
                "actor_q_values": q_values.detach(),
                "target_q_values": target_q_values,
                "q_values": q_values,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_actions = self.model.actor(obs)
        actor_q_values = self.model.q_values(obs, policy_actions)
        actor_terms = _d4pg_loss_terms(
            {
                "logits": logits.detach(),
                "target_distributions": target_distributions.detach(),
                "actor_q_values": actor_q_values,
                "target_q_values": target_q_values.detach(),
                "q_values": q_values.detach(),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "q_value_mean": float(critic_terms["q_value_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
