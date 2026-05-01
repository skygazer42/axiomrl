from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from axiomrl.algorithms._advantage_utils import normalize_advantages
from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.impala import compute_vtrace_targets
from axiomrl.models.mlp_actor_critic import MLPActorCritic


def _appo_loss_terms(
    batch: dict[str, torch.Tensor],
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, torch.Tensor]:
    advantages = normalize_advantages(batch["pg_advantages"])
    log_ratio = batch["target_logprobs"] - batch["behavior_logprobs"]
    ratio = log_ratio.exp()

    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef)) * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    value_loss = 0.5 * F.mse_loss(batch["values"], batch["vtrace_targets"])
    entropy_loss = -batch["entropy"].mean()
    loss = policy_loss + float(vf_coef) * value_loss + float(ent_coef) * entropy_loss

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > float(clip_coef)).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "rho_mean": ratio.mean(),
        "vtrace_target_mean": batch["vtrace_targets"].mean(),
        "pg_advantage_mean": batch["pg_advantages"].mean(),
    }


def appo_loss(
    batch: dict[str, torch.Tensor | float],
    *,
    clip_coef: float,
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
    terms = _appo_loss_terms(
        tensor_batch,
        clip_coef=float(clip_coef),
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
    )
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class APPO:
    def __init__(
        self,
        *,
        policy: MLPActorCritic,
        learning_rate: float,
        clip_coef: float,
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
        if float(clip_coef) < 0.0:
            raise ValueError(f"clip_coef must be >= 0, got {clip_coef}")
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
        self.clip_coef = float(clip_coef)
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

        if obs.ndim != 3:
            raise ValueError(f"expected obs to have shape [T, B, obs_dim], got {tuple(obs.shape)!r}")
        if actions.ndim != 2:
            raise ValueError(f"expected actions to have shape [T, B], got {tuple(actions.shape)!r}")

        time_steps, batch_size = actions.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs.shape[-1])
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

        terms = _appo_loss_terms(
            {
                "target_logprobs": target_logprobs,
                "behavior_logprobs": behavior_logprobs,
                "vtrace_targets": vtrace_targets,
                "values": values,
                "pg_advantages": pg_advantages,
                "entropy": entropy,
            },
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
