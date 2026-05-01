from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from axiomrl.algorithms._advantage_utils import normalize_advantages
from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.cnn import CNNActorCritic
from axiomrl.models.mlp_actor_critic import MLPActorCritic


def compute_vtrace_targets(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    target_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    gamma: float,
    rho_clip: float,
    c_clip: float,
    pg_rho_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rewards.ndim != 2:
        raise ValueError(f"expected rewards to have shape [T, B], got {tuple(rewards.shape)!r}")
    if dones.shape != rewards.shape or values.shape != rewards.shape:
        raise ValueError("rewards, dones, and values must have the same shape")
    if target_logprobs.shape != rewards.shape or behavior_logprobs.shape != rewards.shape:
        raise ValueError("target_logprobs and behavior_logprobs must match rewards shape")

    time_steps, batch_size = rewards.shape
    bootstrap = bootstrap_value.reshape(batch_size).to(dtype=torch.float32, device=rewards.device)

    rhos = torch.exp(target_logprobs - behavior_logprobs)
    clipped_rhos = torch.clamp(rhos, max=float(rho_clip))
    clipped_cs = torch.clamp(rhos, max=float(c_clip))
    clipped_pg_rhos = torch.clamp(rhos, max=float(pg_rho_clip))

    vtrace_targets = torch.zeros_like(values)
    next_vs = bootstrap
    for step in range(time_steps - 1, -1, -1):
        next_non_terminal = 1.0 - dones[step]
        next_values = bootstrap if step == time_steps - 1 else values[step + 1]
        delta = clipped_rhos[step] * (rewards[step] + float(gamma) * next_non_terminal * next_values - values[step])
        vtrace_targets[step] = (
            values[step] + delta + float(gamma) * next_non_terminal * clipped_cs[step] * (next_vs - next_values)
        )
        next_vs = vtrace_targets[step]

    pg_advantages = torch.zeros_like(values)
    for step in range(time_steps):
        next_non_terminal = 1.0 - dones[step]
        next_vs_for_pg = bootstrap if step == time_steps - 1 else vtrace_targets[step + 1]
        pg_advantages[step] = clipped_pg_rhos[step] * (
            rewards[step] + float(gamma) * next_non_terminal * next_vs_for_pg - values[step]
        )

    return vtrace_targets, pg_advantages, rhos


def _impala_loss_terms(
    batch: dict[str, torch.Tensor],
    *,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, torch.Tensor]:
    advantages = normalize_advantages(batch["pg_advantages"])
    policy_loss = -(batch["target_logprobs"] * advantages).mean()
    value_loss = 0.5 * F.mse_loss(batch["values"], batch["vtrace_targets"])
    entropy_loss = -batch["entropy"].mean()
    loss = policy_loss + float(vf_coef) * value_loss + float(ent_coef) * entropy_loss
    rho = torch.exp(batch["target_logprobs"] - batch["behavior_logprobs"])

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "rho_mean": rho.mean(),
        "vtrace_target_mean": batch["vtrace_targets"].mean(),
        "pg_advantage_mean": batch["pg_advantages"].mean(),
    }


def impala_loss(
    batch: dict[str, torch.Tensor | float],
    *,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, float]:
    tensor_batch = {
        "target_logprobs": torch.as_tensor(batch["target_logprobs"], dtype=torch.float32),
        "behavior_logprobs": torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32),
        "vtrace_targets": torch.as_tensor(batch["vtrace_targets"], dtype=torch.float32),
        "values": torch.as_tensor(batch["values"], dtype=torch.float32),
        "pg_advantages": torch.as_tensor(batch["pg_advantages"], dtype=torch.float32),
        "entropy": torch.as_tensor(batch["entropy"], dtype=torch.float32),
    }
    terms = _impala_loss_terms(tensor_batch, ent_coef=float(ent_coef), vf_coef=float(vf_coef))
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class IMPALA:
    def __init__(
        self,
        *,
        policy: MLPActorCritic | CNNActorCritic,
        learning_rate: float,
        ent_coef: float,
        vf_coef: float,
        gamma: float,
        rho_clip: float = 1.0,
        c_clip: float = 1.0,
        pg_rho_clip: float = 1.0,
        max_grad_norm: float = 0.5,
    ) -> None:
        if float(learning_rate) <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
        if float(rho_clip) <= 0.0:
            raise ValueError(f"rho_clip must be > 0, got {rho_clip}")
        if float(c_clip) <= 0.0:
            raise ValueError(f"c_clip must be > 0, got {c_clip}")
        if float(pg_rho_clip) <= 0.0:
            raise ValueError(f"pg_rho_clip must be > 0, got {pg_rho_clip}")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")

        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.gamma = float(gamma)
        self.rho_clip = float(rho_clip)
        self.c_clip = float(c_clip)
        self.pg_rho_clip = float(pg_rho_clip)
        self.max_grad_norm = float(max_grad_norm)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=obs.device)
        behavior_logprobs = torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32, device=obs.device)
        bootstrap_value = torch.as_tensor(batch["bootstrap_value"], dtype=torch.float32, device=obs.device)

        if obs.ndim not in (3, 5):
            raise ValueError(f"expected obs to have shape [T, B, obs_dim] or [T, B, C, H, W], got {tuple(obs.shape)!r}")
        if actions.ndim != 2:
            raise ValueError(f"expected actions to have shape [T, B], got {tuple(actions.shape)!r}")

        time_steps, batch_size = actions.shape
        flat_obs = obs.reshape(time_steps * batch_size, *obs.shape[2:])
        flat_actions = actions.reshape(time_steps * batch_size)
        evaluated = self.policy.evaluate_actions(flat_obs, flat_actions)
        target_logprobs = evaluated["logprobs"].reshape(time_steps, batch_size)
        entropy = evaluated["entropy"].reshape(time_steps, batch_size)
        values = evaluated["values"].reshape(time_steps, batch_size)

        with torch.no_grad():
            vtrace_targets, pg_advantages, _ = compute_vtrace_targets(
                rewards=rewards,
                dones=dones,
                values=values.detach(),
                bootstrap_value=bootstrap_value,
                target_logprobs=target_logprobs.detach(),
                behavior_logprobs=behavior_logprobs,
                gamma=self.gamma,
                rho_clip=self.rho_clip,
                c_clip=self.c_clip,
                pg_rho_clip=self.pg_rho_clip,
            )

        terms = _impala_loss_terms(
            {
                "target_logprobs": target_logprobs,
                "behavior_logprobs": behavior_logprobs,
                "vtrace_targets": vtrace_targets,
                "values": values,
                "pg_advantages": pg_advantages,
                "entropy": entropy,
            },
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
