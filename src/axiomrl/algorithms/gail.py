from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.ppo import PPO
from axiomrl.models.cnn.actor_critic import CNNActorCritic
from axiomrl.models.mlp_actor_critic import MLPActorCritic
from axiomrl.models.mlp_gail_discriminator import CNNGAILDiscriminator, MLPGAILDiscriminator


class GAIL:
    def __init__(
        self,
        *,
        policy: MLPActorCritic | CNNActorCritic,
        discriminator: MLPGAILDiscriminator | CNNGAILDiscriminator,
        learning_rate: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        discriminator_learning_rate: float,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.ppo = PPO(
            policy=policy,
            learning_rate=float(learning_rate),
            clip_coef=float(clip_coef),
            ent_coef=float(ent_coef),
            vf_coef=float(vf_coef),
            max_grad_norm=float(max_grad_norm),
        )
        self.discriminator = discriminator
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(discriminator_learning_rate),
            weight_decay=0.0,
        )

    def discriminator_logits(self, obs: object, actions: object) -> torch.Tensor:
        return self.discriminator(obs, actions)

    def discriminator_reward(self, obs: object, actions: object) -> torch.Tensor:
        logits = self.discriminator_logits(obs, actions)
        return F.softplus(logits)

    def update_discriminator(
        self,
        policy_batch: dict[str, Any],
        expert_batch: dict[str, Any],
        *,
        global_step: int,
    ) -> UpdateResult:
        del global_step
        self.set_train_mode()

        policy_obs = torch.as_tensor(policy_batch["obs"], dtype=torch.float32)
        policy_actions = torch.as_tensor(policy_batch["actions"], dtype=torch.int64, device=policy_obs.device)
        expert_obs = torch.as_tensor(expert_batch["obs"], dtype=torch.float32, device=policy_obs.device)
        expert_actions = torch.as_tensor(expert_batch["actions"], dtype=torch.int64, device=policy_obs.device)

        policy_logits = self.discriminator(policy_obs, policy_actions)
        expert_logits = self.discriminator(expert_obs, expert_actions)

        expert_labels = torch.ones_like(expert_logits)
        policy_labels = torch.zeros_like(policy_logits)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_labels)
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, policy_labels)
        loss = expert_loss + policy_loss

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            expert_probs = torch.sigmoid(expert_logits)
            policy_probs = torch.sigmoid(policy_logits)
            expert_accuracy = (expert_probs >= 0.5).float().mean()
            policy_accuracy = (policy_probs < 0.5).float().mean()

        metrics = {
            "gail_discriminator_loss": float(loss.detach().cpu().item()),
            "gail_discriminator_expert_loss": float(expert_loss.detach().cpu().item()),
            "gail_discriminator_policy_loss": float(policy_loss.detach().cpu().item()),
            "gail_discriminator_expert_accuracy": float(expert_accuracy.detach().cpu().item()),
            "gail_discriminator_policy_accuracy": float(policy_accuracy.detach().cpu().item()),
            "gail_discriminator_expert_logit_mean": float(expert_logits.mean().detach().cpu().item()),
            "gail_discriminator_policy_logit_mean": float(policy_logits.mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        return self.ppo.update(batch, global_step=global_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ppo": self.ppo.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.ppo.load_state_dict(state_dict["ppo"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

    def set_train_mode(self) -> None:
        self.ppo.set_train_mode()
        self.discriminator.train(True)

    def set_eval_mode(self) -> None:
        self.ppo.set_eval_mode()
        self.discriminator.eval()
