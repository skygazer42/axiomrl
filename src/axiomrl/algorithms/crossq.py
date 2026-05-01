from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_crossq import MLPCrossQModel


def _crossq_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"],
        batch["target_q_values"],
    )
    min_sampled_q = torch.minimum(batch["sampled_q1"], batch["sampled_q2"])
    entropy_term = batch["alpha"] * batch["sampled_logprobs"].mean()
    actor_loss = (batch["alpha"] * batch["sampled_logprobs"] - min_sampled_q).mean()

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
        "entropy_term": entropy_term,
    }


def crossq_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "q1_values": torch.as_tensor(batch["q1_values"], dtype=torch.float32),
        "q2_values": torch.as_tensor(batch["q2_values"], dtype=torch.float32),
        "target_q_values": torch.as_tensor(batch["target_q_values"], dtype=torch.float32),
        "sampled_logprobs": torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32),
        "sampled_q1": torch.as_tensor(batch["sampled_q1"], dtype=torch.float32),
        "sampled_q2": torch.as_tensor(batch["sampled_q2"], dtype=torch.float32),
        "alpha": torch.as_tensor(batch["alpha"], dtype=torch.float32),
    }
    terms = _crossq_loss_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class CrossQ:
    def __init__(
        self,
        *,
        model: MLPCrossQModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        policy_delay: int = 3,
        adam_beta1: float = 0.5,
    ) -> None:
        if int(policy_delay) < 1:
            raise ValueError(f"policy_delay must be >= 1, got {policy_delay}")
        if not 0.0 <= float(adam_beta1) < 1.0:
            raise ValueError(f"adam_beta1 must be in [0, 1), got {adam_beta1}")
        self.model = model
        self.policy = model
        betas = (adam_beta1, 0.999)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=learning_rate, betas=betas, weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=learning_rate, betas=betas, weight_decay=0.0
        )
        self.gamma = gamma
        self.alpha = alpha
        self.policy_delay = policy_delay
        self.update_count = 0

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
        batch_size = int(obs.shape[0])

        with torch.no_grad():
            next_policy = self.model.sample_actions(next_obs)

        # CrossQ removes target critics and instead normalizes current and next
        # critic inputs together so both halves share the same batch statistics.
        self.model.set_critic_bn_training_mode(True)
        joint_q1, joint_q2 = self.model.q_values(
            torch.cat([obs, next_obs], dim=0),
            torch.cat([actions, next_policy.actions], dim=0),
        )
        current_q1, next_q1 = joint_q1.split(batch_size, dim=0)
        current_q2, next_q2 = joint_q2.split(batch_size, dim=0)
        target_q_values = rewards + self.gamma * (1.0 - dones) * (
            torch.minimum(next_q1, next_q2).detach() - self.alpha * next_policy.logprobs
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _crossq_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "sampled_logprobs": torch.zeros_like(target_q_values),
                "sampled_q1": torch.zeros_like(target_q_values),
                "sampled_q2": torch.zeros_like(target_q_values),
                "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=target_q_values.device),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        actor_loss_value = torch.zeros((), dtype=torch.float32, device=obs.device)
        entropy_term_value = torch.zeros((), dtype=torch.float32, device=obs.device)
        self.update_count += 1

        if self.update_count % self.policy_delay == 0:
            sampled = self.model.sample_actions(obs)
            self.model.set_critic_bn_training_mode(False)
            sampled_q1, sampled_q2 = self.model.q_values(obs, sampled.actions)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_terms = _crossq_loss_terms(
                {
                    "q1_values": current_q1.detach(),
                    "q2_values": current_q2.detach(),
                    "target_q_values": target_q_values.detach(),
                    "sampled_logprobs": sampled.logprobs,
                    "sampled_q1": sampled_q1,
                    "sampled_q2": sampled_q2,
                    "alpha": torch.as_tensor(self.alpha, dtype=torch.float32, device=sampled.logprobs.device),
                }
            )
            actor_terms["actor_loss"].backward()
            self.actor_optimizer.step()
            actor_loss_value = actor_terms["actor_loss"].detach()
            entropy_term_value = actor_terms["entropy_term"].detach()
            self.model.set_critic_bn_training_mode(True)

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_loss_value.cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "entropy_term": float(entropy_term_value.cpu().item()),
            "policy_delay": float(self.policy_delay),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.update_count = int(state_dict.get("update_count", 0))

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.model.set_critic_bn_training_mode(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
