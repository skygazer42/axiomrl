from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.dreamer import Dreamer
from rl_training.models.dreamer import DreamerModel


class _DiamondDenoiser(nn.Module):
    def __init__(self, *, channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, prior: torch.Tensor, noisy_target: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([prior, noisy_target], dim=1))


class Diamond(Dreamer):
    def __init__(
        self,
        *,
        model: DreamerModel,
        world_model_learning_rate: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        entropy_coef: float,
        denoising_loss_coef: float = 0.5,
        noise_scale: float = 0.15,
        denoiser_hidden_channels: int = 64,
    ) -> None:
        super().__init__(
            model=model,
            world_model_learning_rate=world_model_learning_rate,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        if float(denoising_loss_coef) < 0.0:
            raise ValueError(f"denoising_loss_coef must be >= 0, got {denoising_loss_coef}")
        if float(noise_scale) < 0.0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")

        channels = int(model.obs_shape[0])
        device = next(model.parameters()).device
        self.denoising_loss_coef = float(denoising_loss_coef)
        self.noise_scale = float(noise_scale)
        self.denoiser = _DiamondDenoiser(
            channels=channels,
            hidden_channels=max(8, int(denoiser_hidden_channels)),
        ).to(device)
        self.denoiser_optimizer = torch.optim.Adam(
            self.denoiser.parameters(),
            lr=float(world_model_learning_rate),
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

        if self.noise_scale > 0.0:
            noise = torch.randn_like(obs_target) * self.noise_scale
            noisy_target = (obs_target + noise).clamp(0.0, 1.0)
        else:
            noisy_target = obs_target
        denoised_next = self.denoiser(decoded_next, noisy_target)
        denoising_loss = F.mse_loss(denoised_next, obs_target)

        loss = reconstruction_loss + reward_loss + self.denoising_loss_coef * denoising_loss
        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.denoiser_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.world_model_optimizer.step()
        self.denoiser_optimizer.step()

        metrics = {
            "diamond_world_model_loss": float(loss.detach().cpu().item()),
            "diamond_reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
            "diamond_reward_loss": float(reward_loss.detach().cpu().item()),
            "diamond_denoising_loss": float(denoising_loss.detach().cpu().item()),
            "diamond_denoising_loss_coef": self.denoising_loss_coef,
            "diamond_noise_scale": self.noise_scale,
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["denoiser"] = self.denoiser.state_dict()
        state["denoiser_optimizer"] = self.denoiser_optimizer.state_dict()
        state["denoising_loss_coef"] = self.denoising_loss_coef
        state["noise_scale"] = self.noise_scale
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "denoiser" in state_dict:
            self.denoiser.load_state_dict(state_dict["denoiser"])
        if "denoiser_optimizer" in state_dict:
            self.denoiser_optimizer.load_state_dict(state_dict["denoiser_optimizer"])
        self.denoising_loss_coef = float(state_dict.get("denoising_loss_coef", self.denoising_loss_coef))
        self.noise_scale = float(state_dict.get("noise_scale", self.noise_scale))

    def set_train_mode(self) -> None:
        super().set_train_mode()
        self.denoiser.train(True)

    def set_eval_mode(self) -> None:
        super().set_eval_mode()
        self.denoiser.eval()
