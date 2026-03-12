from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
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


def _flatten_tensors(tensors: Sequence[torch.Tensor], params: Sequence[nn.Parameter]) -> torch.Tensor:
    flat_parts: list[torch.Tensor] = []
    for tensor, param in zip(tensors, params, strict=True):
        if tensor is None:
            flat_parts.append(torch.zeros_like(param).reshape(-1))
        else:
            flat_parts.append(tensor.reshape(-1))
    if not flat_parts:
        raise ValueError("expected at least one parameter tensor to flatten")
    return torch.cat(flat_parts)


def _flat_parameters(params: Sequence[nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in params])


def _set_flat_parameters(params: Sequence[nn.Parameter], flat_params: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.data.copy_(flat_params[offset : offset + numel].view_as(param))
        offset += numel


def _conjugate_gradient(
    fisher_vector_product,
    b: torch.Tensor,
    *,
    num_iterations: int,
    residual_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, int]:
    x = torch.zeros_like(b)
    residual = b.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)

    if residual_dot <= residual_tolerance:
        return x, 0

    for iteration in range(num_iterations):
        fisher_direction = fisher_vector_product(direction)
        denom = torch.dot(direction, fisher_direction)
        if torch.abs(denom) < 1e-8:
            return x, iteration

        alpha = residual_dot / (denom + 1e-8)
        x = x + alpha * direction
        residual = residual - alpha * fisher_direction
        new_residual_dot = torch.dot(residual, residual)
        if new_residual_dot <= residual_tolerance:
            return x, iteration + 1
        beta = new_residual_dot / (residual_dot + 1e-8)
        direction = residual + beta * direction
        residual_dot = new_residual_dot

    return x, num_iterations


def _trpo_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    old_logprobs = batch.get("old_logprobs", batch["logprobs"])
    advantages = _normalize_advantages(batch["advantages"])
    log_ratio = batch["new_logprobs"] - old_logprobs
    ratio = log_ratio.exp()

    surrogate_gain = (ratio * advantages).mean()
    policy_loss = -surrogate_gain
    value_loss = 0.5 * F.mse_loss(batch["values"], batch["returns"])
    entropy = batch["entropy"].mean()
    approx_kl = torch.as_tensor(
        batch.get("approx_kl", torch.zeros((), device=surrogate_gain.device)),
        dtype=torch.float32,
        device=surrogate_gain.device,
    )
    accepted_step = torch.as_tensor(
        batch.get("accepted_step", torch.zeros((), device=surrogate_gain.device)),
        dtype=torch.float32,
        device=surrogate_gain.device,
    )
    step_norm = torch.as_tensor(
        batch.get("step_norm", torch.zeros((), device=surrogate_gain.device)),
        dtype=torch.float32,
        device=surrogate_gain.device,
    )
    backtrack_steps = torch.as_tensor(
        batch.get("backtrack_steps", torch.zeros((), device=surrogate_gain.device)),
        dtype=torch.float32,
        device=surrogate_gain.device,
    )
    cg_iterations = torch.as_tensor(
        batch.get("cg_iterations", torch.zeros((), device=surrogate_gain.device)),
        dtype=torch.float32,
        device=surrogate_gain.device,
    )

    return {
        "surrogate_gain": surrogate_gain,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "accepted_step": accepted_step,
        "step_norm": step_norm,
        "backtrack_steps": backtrack_steps,
        "cg_iterations": cg_iterations,
    }


def trpo_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _trpo_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class TRPO:
    def __init__(
        self,
        *,
        policy: MLPActorCritic,
        value_learning_rate: float,
        max_kl: float,
        cg_iterations: int = 10,
        cg_damping: float = 1e-1,
        line_search_steps: int = 10,
        line_search_shrink: float = 0.8,
        value_updates: int = 5,
        ent_coef: float = 0.0,
    ) -> None:
        if float(value_learning_rate) <= 0.0:
            raise ValueError(f"value_learning_rate must be > 0, got {value_learning_rate}")
        if float(max_kl) <= 0.0:
            raise ValueError(f"max_kl must be > 0, got {max_kl}")
        if int(cg_iterations) < 1:
            raise ValueError(f"cg_iterations must be >= 1, got {cg_iterations}")
        if float(cg_damping) < 0.0:
            raise ValueError(f"cg_damping must be >= 0, got {cg_damping}")
        if int(line_search_steps) < 1:
            raise ValueError(f"line_search_steps must be >= 1, got {line_search_steps}")
        if not 0.0 < float(line_search_shrink) < 1.0:
            raise ValueError(f"line_search_shrink must be in (0, 1), got {line_search_shrink}")
        if int(value_updates) < 1:
            raise ValueError(f"value_updates must be >= 1, got {value_updates}")
        if float(ent_coef) < 0.0:
            raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")

        self.policy = policy
        self.value_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=value_learning_rate)
        self.max_kl = float(max_kl)
        self.cg_iterations = int(cg_iterations)
        self.cg_damping = float(cg_damping)
        self.line_search_steps = int(line_search_steps)
        self.line_search_shrink = float(line_search_shrink)
        self.value_updates = int(value_updates)
        self.ent_coef = float(ent_coef)

    def _evaluate_actor(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_dist: Categorical,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = Categorical(logits=self.policy.actor(obs))
        new_logprobs = distribution.log_prob(actions)
        ratio = (new_logprobs - old_logprobs).exp()
        surrogate_gain = (ratio * advantages).mean()
        entropy = distribution.entropy().mean()
        objective = surrogate_gain + self.ent_coef * entropy
        approx_kl = kl_divergence(old_dist, distribution).mean()
        return objective, surrogate_gain, entropy, approx_kl, new_logprobs

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        device = next(self.policy.parameters()).device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
        advantages = _normalize_advantages(torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device))
        old_logprobs = torch.as_tensor(batch.get("old_logprobs", batch["logprobs"]), dtype=torch.float32, device=device)

        actor_params = tuple(self.policy.actor.parameters())
        with torch.no_grad():
            old_dist = Categorical(logits=self.policy.actor(obs).detach())

        objective, old_surrogate, _, _, _ = self._evaluate_actor(obs, actions, old_dist, old_logprobs, advantages)
        flat_policy_grad = _flatten_tensors(
            torch.autograd.grad(objective, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        ).detach()

        accepted_step = 0.0
        accepted_fraction = 0.0
        backtrack_steps = float(self.line_search_steps)
        step_norm = 0.0
        cg_iteration_count = 0

        if torch.isfinite(flat_policy_grad).all() and torch.norm(flat_policy_grad) > 1e-8:

            def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
                _, _, _, approx_kl, _ = self._evaluate_actor(obs, actions, old_dist, old_logprobs, advantages)
                flat_kl_grad = _flatten_tensors(
                    torch.autograd.grad(approx_kl, actor_params, create_graph=True, allow_unused=True),
                    actor_params,
                )
                directional_kl = torch.dot(flat_kl_grad, vector)
                hessian_vector = _flatten_tensors(
                    torch.autograd.grad(directional_kl, actor_params, allow_unused=True),
                    actor_params,
                ).detach()
                return hessian_vector + self.cg_damping * vector

            step_direction, cg_iteration_count = _conjugate_gradient(
                fisher_vector_product,
                flat_policy_grad,
                num_iterations=self.cg_iterations,
            )
            fisher_step = fisher_vector_product(step_direction)
            step_curvature = torch.dot(step_direction, fisher_step)

            if torch.isfinite(step_curvature) and step_curvature > 1e-8:
                full_step = step_direction * torch.sqrt(
                    torch.as_tensor(2.0 * self.max_kl, dtype=torch.float32, device=device) / (step_curvature + 1e-8)
                )
                old_actor_params = _flat_parameters(actor_params)
                old_objective = objective.detach()

                for backtrack_index in range(self.line_search_steps):
                    fraction = self.line_search_shrink**backtrack_index
                    candidate_step = fraction * full_step
                    _set_flat_parameters(actor_params, old_actor_params + candidate_step)

                    with torch.no_grad():
                        candidate_objective, _, _, candidate_kl, _ = self._evaluate_actor(
                            obs,
                            actions,
                            old_dist,
                            old_logprobs,
                            advantages,
                        )

                    if (
                        torch.isfinite(candidate_objective)
                        and torch.isfinite(candidate_kl)
                        and candidate_objective > old_objective
                        and candidate_kl <= self.max_kl
                    ):
                        accepted_step = 1.0
                        accepted_fraction = float(fraction)
                        backtrack_steps = float(backtrack_index)
                        step_norm = float(candidate_step.norm().detach().cpu().item())
                        break

                if not accepted_step:
                    _set_flat_parameters(actor_params, old_actor_params)

        final_value_loss = torch.zeros((), dtype=torch.float32, device=device)
        for _ in range(self.value_updates):
            values = self.policy.critic(obs).squeeze(-1)
            final_value_loss = 0.5 * F.mse_loss(values, returns)
            self.value_optimizer.zero_grad(set_to_none=True)
            final_value_loss.backward()
            self.value_optimizer.step()

        with torch.no_grad():
            distribution = Categorical(logits=self.policy.actor(obs))
            new_logprobs = distribution.log_prob(actions)
            entropy = distribution.entropy()
            approx_kl = kl_divergence(old_dist, distribution).mean()
            values = self.policy.critic(obs).squeeze(-1)

        metrics = trpo_loss(
            {
                "logprobs": old_logprobs,
                "new_logprobs": new_logprobs,
                "advantages": advantages,
                "returns": returns,
                "values": values,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "accepted_step": torch.as_tensor(accepted_step, dtype=torch.float32, device=device),
                "step_norm": torch.as_tensor(step_norm, dtype=torch.float32, device=device),
                "backtrack_steps": torch.as_tensor(backtrack_steps, dtype=torch.float32, device=device),
                "cg_iterations": torch.as_tensor(float(cg_iteration_count), dtype=torch.float32, device=device),
            }
        )
        metrics["line_search_fraction"] = accepted_fraction
        metrics["value_updates"] = float(self.value_updates)
        metrics["final_value_loss"] = float(final_value_loss.detach().cpu().item())
        metrics["surrogate_improvement"] = float(
            (metrics["surrogate_gain"] - old_surrogate.detach().cpu().item())
        )
        return UpdateResult(metrics=metrics, num_gradient_steps=1 + self.value_updates)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])

    def set_train_mode(self) -> None:
        self.policy.train(True)

    def set_eval_mode(self) -> None:
        self.policy.eval()
