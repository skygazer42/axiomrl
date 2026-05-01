from typing import Any

import torch
from torch import nn

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.recurrent.lstm_actor_critic import LSTMActorCritic

__all__ = ["RecurrentPPOAlgorithm"]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=values.dtype, device=values.device)
    denominator = mask_tensor.sum().clamp_min(1.0)
    return (values * mask_tensor).sum() / denominator


def _normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=advantages.dtype, device=advantages.device)
    denominator = mask_tensor.sum().clamp_min(1.0)
    mean = (advantages * mask_tensor).sum() / denominator
    variance = ((advantages - mean) ** 2 * mask_tensor).sum() / denominator
    std = variance.sqrt()
    if std < 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


class RecurrentPPOAlgorithm:
    def __init__(
        self,
        *,
        policy: LSTMActorCritic,
        learning_rate: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=0.0)
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        episode_starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32)
        initial_state = (
            torch.as_tensor(batch["initial_h"], dtype=torch.float32),
            torch.as_tensor(batch["initial_c"], dtype=torch.float32),
        )

        evaluated = self.policy.evaluate_actions_sequence(
            obs,
            actions,
            initial_state=initial_state,
            episode_starts=episode_starts,
        )
        normalized_advantages = _normalize_advantages(advantages, mask)
        log_ratio = evaluated["logprobs"] - old_logprobs
        ratio = log_ratio.exp()
        unclipped_objective = ratio * normalized_advantages
        clipped_objective = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * normalized_advantages

        policy_loss = -_masked_mean(torch.minimum(unclipped_objective, clipped_objective), mask)
        value_loss = 0.5 * _masked_mean((evaluated["values"] - returns) ** 2, mask)
        entropy_loss = -_masked_mean(evaluated["entropy"], mask)
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
        clip_fraction = _masked_mean(((ratio - 1.0).abs() > self.clip_coef).float(), mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy_loss": float(entropy_loss.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
        }
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
