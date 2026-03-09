from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_actor_critic import MLPActorCritic


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages

    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    if std < 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


def _ppo_loss_terms(
    minibatch: dict[str, torch.Tensor],
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, torch.Tensor]:
    old_logprobs = minibatch.get("old_logprobs", minibatch["logprobs"])
    advantages = _normalize_advantages(minibatch["advantages"])
    log_ratio = minibatch["new_logprobs"] - old_logprobs
    ratio = log_ratio.exp()

    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    value_loss = 0.5 * F.mse_loss(minibatch["new_values"], minibatch["returns"])
    entropy_loss = -minibatch["entropy"].mean()
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def ppo_loss(
    minibatch: dict[str, torch.Tensor],
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, float]:
    terms = _ppo_loss_terms(
        minibatch,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class PPO:
    def __init__(
        self,
        *,
        policy: MLPActorCritic,
        learning_rate: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        evaluated = self.policy.evaluate_actions(obs, actions)

        minibatch = {
            "logprobs": torch.as_tensor(batch["logprobs"], dtype=torch.float32),
            "advantages": torch.as_tensor(batch["advantages"], dtype=torch.float32),
            "returns": torch.as_tensor(batch["returns"], dtype=torch.float32),
            "new_logprobs": evaluated["logprobs"],
            "new_values": evaluated["values"],
            "entropy": evaluated["entropy"],
        }

        terms = _ppo_loss_terms(
            minibatch,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
        )

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.policy.train(True)

    def set_eval_mode(self) -> None:
        self.policy.eval()
