import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.cnn.drqv2 import CNNDrQv2Model


def _random_crop(obs: torch.Tensor, *, pad: int) -> torch.Tensor:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(0)
    if pad <= 0:
        return obs_tensor

    batch_size, _, height, width = obs_tensor.shape
    padded = F.pad(obs_tensor, (pad, pad, pad, pad), mode="replicate")
    max_offset = pad * 2 + 1
    offsets_y = torch.randint(0, max_offset, (batch_size,), device=obs_tensor.device)
    offsets_x = torch.randint(0, max_offset, (batch_size,), device=obs_tensor.device)

    cropped = [
        padded[
            index : index + 1,
            :,
            int(offset_y.item()) : int(offset_y.item()) + height,
            int(offset_x.item()) : int(offset_x.item()) + width,
        ]
        for index, (offset_y, offset_x) in enumerate(zip(offsets_y, offsets_x, strict=True))
    ]
    return torch.cat(cropped, dim=0)


def _drqv2_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    critic_loss = F.mse_loss(batch["q1_values"], batch["target_q_values"]) + F.mse_loss(
        batch["q2_values"], batch["target_q_values"]
    )
    actor_loss = -batch["actor_q_values"].mean()
    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "target_q_mean": batch["target_q_values"].mean(),
    }


def drqv2_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _drqv2_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class DrQv2:
    def __init__(
        self,
        *,
        model: CNNDrQv2Model,
        learning_rate: float,
        gamma: float,
        tau: float,
        policy_delay: int,
        augmentation_pad: int,
    ) -> None:
        # This v1 keeps the random-shift augmentation path explicit while
        # reusing the package's existing single-process TD3-style learner loop.
        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=learning_rate, weight_decay=0.0)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.augmentation_pad = augmentation_pad
        self.update_count = 0

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        augmented_obs = _random_crop(obs, pad=self.augmentation_pad)
        augmented_next_obs = _random_crop(next_obs, pad=self.augmentation_pad)
        current_q1, current_q2 = self.model.q_values(augmented_obs, actions)

        with torch.no_grad():
            next_actions = self.target_model.actor(augmented_next_obs)
            target_q1, target_q2 = self.target_model.q_values(augmented_next_obs, next_actions)
            target_q_values = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _drqv2_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "actor_q_values": current_q1.detach(),
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        actor_loss_value = torch.tensor(0.0, dtype=torch.float32, device=obs.device)
        self.update_count += 1

        if self.update_count % self.policy_delay == 0:
            actor_obs = _random_crop(obs, pad=self.augmentation_pad)
            self.actor_optimizer.zero_grad(set_to_none=True)
            policy_actions = self.model.actor(actor_obs)
            actor_q1, _ = self.model.q_values(actor_obs, policy_actions)
            actor_terms = _drqv2_loss_terms(
                {
                    "q1_values": current_q1.detach(),
                    "q2_values": current_q2.detach(),
                    "target_q_values": target_q_values.detach(),
                    "actor_q_values": actor_q1,
                }
            )
            actor_terms["actor_loss"].backward()
            self.actor_optimizer.step()
            self.soft_update_targets()
            actor_loss_value = actor_terms["actor_loss"].detach()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "actor_loss": float(actor_loss_value.cpu().item()),
            "target_q_mean": float(critic_terms["target_q_mean"].detach().cpu().item()),
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
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.update_count = int(state_dict.get("update_count", 0))

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
