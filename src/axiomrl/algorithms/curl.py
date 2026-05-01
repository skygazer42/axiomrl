import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.drq import _random_crop
from axiomrl.models.cnn.curl import CNNCURLModel


def _critic_loss_terms(
    *,
    q1_values: torch.Tensor,
    q2_values: torch.Tensor,
    target_q_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    return {
        "critic_loss": critic_loss,
        "target_q_mean": target_q_values.mean(),
    }


def _actor_loss_terms(
    *,
    sampled_logprobs: torch.Tensor,
    sampled_q1: torch.Tensor,
    sampled_q2: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, torch.Tensor]:
    entropy_term = alpha * sampled_logprobs.mean()
    actor_loss = (alpha * sampled_logprobs - torch.minimum(sampled_q1, sampled_q2)).mean()
    return {
        "actor_loss": actor_loss,
        "entropy_term": entropy_term,
    }


def _contrastive_logits(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    return torch.matmul(query_embeddings, key_embeddings.T) / temperature


def _contrastive_loss_terms(
    *,
    curl_logits: torch.Tensor,
    curl_labels: torch.Tensor,
    curl_coef: torch.Tensor,
    critic_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    curl_loss = F.cross_entropy(curl_logits, curl_labels)
    total_critic_loss = critic_loss + curl_coef * curl_loss
    return {
        "curl_loss": curl_loss,
        "total_critic_loss": total_critic_loss,
    }


def curl_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "q1_values": torch.as_tensor(batch["q1_values"], dtype=torch.float32),
        "q2_values": torch.as_tensor(batch["q2_values"], dtype=torch.float32),
        "target_q_values": torch.as_tensor(batch["target_q_values"], dtype=torch.float32),
        "sampled_logprobs": torch.as_tensor(batch["sampled_logprobs"], dtype=torch.float32),
        "sampled_q1": torch.as_tensor(batch["sampled_q1"], dtype=torch.float32),
        "sampled_q2": torch.as_tensor(batch["sampled_q2"], dtype=torch.float32),
        "alpha": torch.as_tensor(batch["alpha"], dtype=torch.float32),
        "curl_logits": torch.as_tensor(batch["curl_logits"], dtype=torch.float32),
        "curl_labels": torch.as_tensor(batch["curl_labels"], dtype=torch.int64),
        "curl_coef": torch.as_tensor(batch["curl_coef"], dtype=torch.float32),
    }
    critic_terms = _critic_loss_terms(
        q1_values=tensor_batch["q1_values"],
        q2_values=tensor_batch["q2_values"],
        target_q_values=tensor_batch["target_q_values"],
    )
    actor_terms = _actor_loss_terms(
        sampled_logprobs=tensor_batch["sampled_logprobs"],
        sampled_q1=tensor_batch["sampled_q1"],
        sampled_q2=tensor_batch["sampled_q2"],
        alpha=tensor_batch["alpha"],
    )
    contrastive_terms = _contrastive_loss_terms(
        curl_logits=tensor_batch["curl_logits"],
        curl_labels=tensor_batch["curl_labels"],
        curl_coef=tensor_batch["curl_coef"],
        critic_loss=critic_terms["critic_loss"],
    )
    metrics = {**critic_terms, **actor_terms, **contrastive_terms}
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


class CURL:
    def __init__(
        self,
        *,
        model: CNNCURLModel,
        learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        augmentation_pad: int,
        curl_temperature: float,
        curl_coef: float,
    ) -> None:
        if curl_temperature <= 0.0:
            raise ValueError("curl_temperature must be positive")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=learning_rate, weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.augmentation_pad = augmentation_pad
        self.curl_temperature = curl_temperature
        self.curl_coef = curl_coef
        self.update_count = 0

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        critic_obs = _random_crop(obs, pad=self.augmentation_pad)
        current_q1, current_q2 = self.model.q_values(critic_obs, actions)

        with torch.no_grad():
            augmented_next_obs = _random_crop(next_obs, pad=self.augmentation_pad)
            next_policy = self.target_model.sample_actions(augmented_next_obs)
            target_q1, target_q2 = self.target_model.q_values(augmented_next_obs, next_policy.actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * (
                torch.minimum(target_q1, target_q2) - self.alpha * next_policy.logprobs
            )
            key_obs = _random_crop(obs, pad=self.augmentation_pad)
            key_embeddings = self.target_model.curl_embeddings(key_obs)

        critic_terms = _critic_loss_terms(
            q1_values=current_q1,
            q2_values=current_q2,
            target_q_values=target_q_values,
        )
        query_obs = _random_crop(obs, pad=self.augmentation_pad)
        query_embeddings = self.model.curl_embeddings(query_obs)
        curl_logits = _contrastive_logits(
            query_embeddings,
            key_embeddings,
            temperature=self.curl_temperature,
        )
        curl_labels = torch.arange(curl_logits.shape[0], device=curl_logits.device)
        contrastive_terms = _contrastive_loss_terms(
            curl_logits=curl_logits,
            curl_labels=curl_labels,
            curl_coef=torch.as_tensor(self.curl_coef, dtype=torch.float32, device=curl_logits.device),
            critic_loss=critic_terms["critic_loss"],
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        contrastive_terms["total_critic_loss"].backward()
        self.critic_optimizer.step()

        actor_obs = _random_crop(obs, pad=self.augmentation_pad)
        sampled = self.model.sample_actions(actor_obs)
        sampled_q1, sampled_q2 = self.model.q_values(actor_obs, sampled.actions)
        actor_terms = _actor_loss_terms(
            sampled_logprobs=sampled.logprobs,
            sampled_q1=sampled_q1,
            sampled_q2=sampled_q2,
            alpha=torch.as_tensor(self.alpha, dtype=torch.float32, device=sampled.logprobs.device),
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()
        self.update_count += 1

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "entropy_term": float(actor_terms["entropy_term"].detach().cpu().item()),
            "curl_loss": float(contrastive_terms["curl_loss"].detach().cpu().item()),
            "total_critic_loss": float(contrastive_terms["total_critic_loss"].detach().cpu().item()),
            "algorithm_updates": float(self.update_count),
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
            "alpha": self.alpha,
            "curl_temperature": self.curl_temperature,
            "curl_coef": self.curl_coef,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.curl_temperature = float(state_dict.get("curl_temperature", self.curl_temperature))
        self.curl_coef = float(state_dict.get("curl_coef", self.curl_coef))
        self.update_count = int(state_dict.get("update_count", 0))

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
