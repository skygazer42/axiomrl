import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_bcq import MLPBCQModel


def _bcq_loss_terms(batch: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    reconstruction_loss = torch.as_tensor(batch["reconstruction_loss"], dtype=torch.float32, device=q1_values.device)
    kl_loss = torch.as_tensor(batch["kl_loss"], dtype=torch.float32, device=q1_values.device)
    vae_kl_weight = torch.as_tensor(batch["vae_kl_weight"], dtype=torch.float32, device=q1_values.device)
    actor_q_values = torch.as_tensor(batch["actor_q_values"], dtype=torch.float32, device=q1_values.device)
    candidate_q_values = torch.as_tensor(batch["candidate_q_values"], dtype=torch.float32, device=q1_values.device)

    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    actor_loss = -actor_q_values.mean()
    vae_loss = reconstruction_loss + vae_kl_weight * kl_loss
    return {
        "vae_loss": vae_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "candidate_q_mean": candidate_q_values.mean(),
    }


def bcq_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    terms = _bcq_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _repeat_observations(obs: torch.Tensor, num_action_samples: int) -> torch.Tensor:
    return obs.unsqueeze(1).expand(-1, num_action_samples, -1).reshape(obs.shape[0] * num_action_samples, obs.shape[1])


def _reshape_candidate_values(values: torch.Tensor, batch_size: int, num_action_samples: int) -> torch.Tensor:
    return values.reshape(batch_size, num_action_samples)


class BCQ:
    def __init__(
        self,
        *,
        model: MLPBCQModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        num_action_samples: int,
        vae_kl_weight: float,
    ) -> None:
        if int(num_action_samples) < 1:
            raise ValueError(f"num_action_samples must be >= 1, got {num_action_samples}")
        if float(vae_kl_weight) < 0.0:
            raise ValueError(f"vae_kl_weight must be >= 0, got {vae_kl_weight}")
        if int(model.latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {model.latent_dim}")
        if float(model.perturbation_scale) <= 0.0:
            raise ValueError(f"perturbation_scale must be > 0, got {model.perturbation_scale}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.vae_optimizer = torch.optim.Adam(self.model.vae_parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.num_action_samples = int(num_action_samples)
        self.vae_kl_weight = float(vae_kl_weight)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        reconstructed_actions, latent_mean, latent_log_std = self.model.reconstruct(obs, actions)
        reconstruction_loss = F.mse_loss(reconstructed_actions, actions)
        latent_variance = torch.exp(2.0 * latent_log_std)
        kl_loss = -0.5 * (1.0 + 2.0 * latent_log_std - latent_mean.pow(2) - latent_variance).sum(dim=-1).mean()

        self.vae_optimizer.zero_grad(set_to_none=True)
        vae_terms = _bcq_loss_terms(
            {
                "q1_values": torch.zeros_like(rewards),
                "q2_values": torch.zeros_like(rewards),
                "target_q_values": torch.zeros_like(rewards),
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "vae_kl_weight": torch.as_tensor(self.vae_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": torch.zeros_like(rewards),
                "candidate_q_values": torch.zeros_like(rewards),
            }
        )
        vae_terms["vae_loss"].backward()
        self.vae_optimizer.step()

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            candidate_next_actions = self.target_model.sample_candidate_actions(
                next_obs,
                num_action_samples=self.num_action_samples,
                deterministic=False,
            )
            flat_next_obs = _repeat_observations(next_obs, self.num_action_samples)
            flat_next_actions = candidate_next_actions.reshape(-1, self.model.action_dim)
            target_q1, target_q2 = self.target_model.q_values(flat_next_obs, flat_next_actions)
            target_q1 = _reshape_candidate_values(target_q1, next_obs.shape[0], self.num_action_samples)
            target_q2 = _reshape_candidate_values(target_q2, next_obs.shape[0], self.num_action_samples)
            candidate_q_values = torch.minimum(target_q1, target_q2)
            next_values = candidate_q_values.max(dim=1).values
            target_q_values = rewards + self.gamma * (1.0 - dones) * next_values

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _bcq_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "reconstruction_loss": reconstruction_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "vae_kl_weight": torch.as_tensor(self.vae_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": torch.zeros_like(rewards),
                "candidate_q_values": candidate_q_values,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        sampled_behavior_actions = self.model.decode(obs, deterministic=False).detach()
        perturbed_actions = self.model.perturb_actions(obs, sampled_behavior_actions)
        actor_q1, actor_q2 = self.model.q_values(obs, perturbed_actions)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _bcq_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "vae_kl_weight": torch.as_tensor(self.vae_kl_weight, dtype=torch.float32, device=obs.device),
                "actor_q_values": actor_q1,
                "candidate_q_values": torch.minimum(actor_q1, actor_q2),
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "vae_loss": float(vae_terms["vae_loss"].detach().cpu().item()),
            "reconstruction_loss": float(vae_terms["reconstruction_loss"].detach().cpu().item()),
            "kl_loss": float(vae_terms["kl_loss"].detach().cpu().item()),
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
            "candidate_q_mean": float(critic_terms["candidate_q_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.vae_optimizer.load_state_dict(state_dict["vae_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
