import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_iql import MLPIQLModel


def gumbel_rescale_loss(
    diff: torch.Tensor | float,
    *,
    alpha: float,
    clip_max: float | None = None,
) -> torch.Tensor:
    alpha_value = float(alpha)
    if alpha_value <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    diff_tensor = torch.as_tensor(diff, dtype=torch.float32)
    z = diff_tensor / alpha_value
    if clip_max is not None:
        z = torch.minimum(z, torch.full_like(z, float(clip_max)))

    max_z = z.max().clamp(min=-1.0).detach()
    exp_neg_max_z = torch.exp(-max_z)
    return torch.exp(z - max_z) - z * exp_neg_max_z - exp_neg_max_z


def _expectile_value_loss(diff: torch.Tensor, *, expectile: float) -> torch.Tensor:
    weight = torch.where(
        diff > 0,
        torch.full_like(diff, float(expectile)),
        torch.full_like(diff, 1.0 - float(expectile)),
    )
    return weight * diff.pow(2)


def _xql_loss_terms(batch: dict[str, torch.Tensor | float | bool | None]) -> dict[str, torch.Tensor]:
    q1_values = torch.as_tensor(batch["q1_values"], dtype=torch.float32)
    q2_values = torch.as_tensor(batch["q2_values"], dtype=torch.float32, device=q1_values.device)
    target_q_values = torch.as_tensor(batch["target_q_values"], dtype=torch.float32, device=q1_values.device)
    value_predictions = torch.as_tensor(batch["value_predictions"], dtype=torch.float32, device=q1_values.device)
    target_state_values = torch.as_tensor(batch["target_state_values"], dtype=torch.float32, device=q1_values.device)
    behavior_logprobs = torch.as_tensor(batch["behavior_logprobs"], dtype=torch.float32, device=q1_values.device)
    advantage_weights = torch.as_tensor(batch["advantage_weights"], dtype=torch.float32, device=q1_values.device)
    loss_temperature = float(batch.get("loss_temperature", 1.0))
    expectile = float(batch.get("expectile", 0.7))
    vanilla_value_loss = bool(batch.get("vanilla_value_loss", False))
    max_value_diff_exp = batch.get("max_value_diff_exp")

    critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
    value_advantage = target_state_values - value_predictions
    if vanilla_value_loss:
        value_loss = _expectile_value_loss(value_advantage, expectile=expectile).mean()
    else:
        clip_max = None if max_value_diff_exp is None else float(max_value_diff_exp)
        value_loss = gumbel_rescale_loss(value_advantage, alpha=loss_temperature, clip_max=clip_max).mean()
    actor_loss = -(advantage_weights * behavior_logprobs).mean()

    return {
        "critic_loss": critic_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "target_q_mean": target_q_values.mean(),
        "advantage_weight_mean": advantage_weights.mean(),
        "value_advantage_mean": value_advantage.mean(),
        "value_prediction_mean": value_predictions.mean(),
        "target_state_value_mean": target_state_values.mean(),
    }


def xql_value_loss(batch: dict[str, torch.Tensor | float | bool | None]) -> dict[str, float]:
    terms = _xql_loss_terms(
        {
            "q1_values": torch.zeros_like(torch.as_tensor(batch["value_predictions"], dtype=torch.float32)),
            "q2_values": torch.zeros_like(torch.as_tensor(batch["value_predictions"], dtype=torch.float32)),
            "target_q_values": torch.zeros_like(torch.as_tensor(batch["value_predictions"], dtype=torch.float32)),
            "value_predictions": batch["value_predictions"],
            "target_state_values": batch["target_state_values"],
            "behavior_logprobs": torch.zeros_like(torch.as_tensor(batch["value_predictions"], dtype=torch.float32)),
            "advantage_weights": torch.ones_like(torch.as_tensor(batch["value_predictions"], dtype=torch.float32)),
            "loss_temperature": batch.get("loss_temperature", 1.0),
            "expectile": batch.get("expectile", 0.7),
            "vanilla_value_loss": batch.get("vanilla_value_loss", False),
            "max_value_diff_exp": batch.get("max_value_diff_exp"),
        }
    )
    return {
        "value_loss": float(terms["value_loss"].detach().cpu().item()),
        "value_advantage_mean": float(terms["value_advantage_mean"].detach().cpu().item()),
        "value_prediction_mean": float(terms["value_prediction_mean"].detach().cpu().item()),
        "target_state_value_mean": float(terms["target_state_value_mean"].detach().cpu().item()),
    }


def xql_loss(batch: dict[str, torch.Tensor | float | bool | None]) -> dict[str, float]:
    terms = _xql_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class XQL:
    def __init__(
        self,
        *,
        model: MLPIQLModel,
        learning_rate: float,
        gamma: float,
        tau: float,
        beta: float,
        loss_temperature: float,
        max_advantage_weight: float,
        vanilla_value_loss: bool = False,
        expectile: float = 0.7,
        max_value_diff_exp: float | None = 5.0,
    ) -> None:
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if float(loss_temperature) <= 0.0:
            raise ValueError(f"loss_temperature must be > 0, got {loss_temperature}")
        if float(max_advantage_weight) <= 0.0:
            raise ValueError(f"max_advantage_weight must be > 0, got {max_advantage_weight}")
        if bool(vanilla_value_loss) and not 0.0 < float(expectile) < 1.0:
            raise ValueError(f"expectile must be in (0, 1), got {expectile}")
        if max_value_diff_exp is not None and float(max_value_diff_exp) <= 0.0:
            raise ValueError(f"max_value_diff_exp must be > 0, got {max_value_diff_exp}")

        self.model = model
        self.policy = model
        self.target_model = copy.deepcopy(model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.value_optimizer = torch.optim.Adam(
            self.model.value_parameters(), lr=float(learning_rate), weight_decay=0.0
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.beta = float(beta)
        self.loss_temperature = float(loss_temperature)
        self.max_advantage_weight = float(max_advantage_weight)
        self.vanilla_value_loss = bool(vanilla_value_loss)
        self.expectile = float(expectile)
        self.max_value_diff_exp = None if max_value_diff_exp is None else float(max_value_diff_exp)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        current_q1, current_q2 = self.model.q_values(obs, actions)

        with torch.no_grad():
            next_values = self.model.value(next_obs)
            target_q_values = rewards + self.gamma * (1.0 - dones) * next_values

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_terms = _xql_loss_terms(
            {
                "q1_values": current_q1,
                "q2_values": current_q2,
                "target_q_values": target_q_values,
                "value_predictions": torch.zeros_like(rewards),
                "target_state_values": torch.zeros_like(rewards),
                "behavior_logprobs": torch.zeros_like(rewards),
                "advantage_weights": torch.ones_like(rewards),
                "loss_temperature": self.loss_temperature,
                "expectile": self.expectile,
                "vanilla_value_loss": self.vanilla_value_loss,
                "max_value_diff_exp": self.max_value_diff_exp,
            }
        )
        critic_terms["critic_loss"].backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            target_q1, target_q2 = self.target_model.q_values(obs, actions)
            target_state_values = torch.minimum(target_q1, target_q2)

        value_predictions = self.model.value(obs)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_terms = _xql_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "value_predictions": value_predictions,
                "target_state_values": target_state_values,
                "behavior_logprobs": torch.zeros_like(rewards),
                "advantage_weights": torch.ones_like(rewards),
                "loss_temperature": self.loss_temperature,
                "expectile": self.expectile,
                "vanilla_value_loss": self.vanilla_value_loss,
                "max_value_diff_exp": self.max_value_diff_exp,
            }
        )
        value_terms["value_loss"].backward()
        self.value_optimizer.step()

        with torch.no_grad():
            advantages = target_state_values - value_predictions.detach()
            advantage_weights = torch.exp(advantages / self.beta).clamp(max=self.max_advantage_weight)

        behavior_logprobs = self.model.action_logprobs(obs, actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_terms = _xql_loss_terms(
            {
                "q1_values": current_q1.detach(),
                "q2_values": current_q2.detach(),
                "target_q_values": target_q_values.detach(),
                "value_predictions": value_predictions.detach(),
                "target_state_values": target_state_values.detach(),
                "behavior_logprobs": behavior_logprobs,
                "advantage_weights": advantage_weights,
                "loss_temperature": self.loss_temperature,
                "expectile": self.expectile,
                "vanilla_value_loss": self.vanilla_value_loss,
                "max_value_diff_exp": self.max_value_diff_exp,
            }
        )
        actor_terms["actor_loss"].backward()
        self.actor_optimizer.step()

        self.soft_update_targets()

        metrics = {
            "critic_loss": float(critic_terms["critic_loss"].detach().cpu().item()),
            "value_loss": float(value_terms["value_loss"].detach().cpu().item()),
            "actor_loss": float(actor_terms["actor_loss"].detach().cpu().item()),
            "target_q_mean": float(actor_terms["target_q_mean"].detach().cpu().item()),
            "advantage_weight_mean": float(actor_terms["advantage_weight_mean"].detach().cpu().item()),
            "value_advantage_mean": float(value_terms["value_advantage_mean"].detach().cpu().item()),
            "value_prediction_mean": float(value_terms["value_prediction_mean"].detach().cpu().item()),
            "target_state_value_mean": float(value_terms["target_state_value_mean"].detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_model.q1.parameters(), self.model.q1.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        for target_param, param in zip(self.target_model.q2.parameters(), self.model.q2.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.value_optimizer.load_state_dict(state_dict["value_optimizer"])

    def set_train_mode(self) -> None:
        self.model.train(True)
        self.target_model.train(False)

    def set_eval_mode(self) -> None:
        self.model.eval()
        self.target_model.eval()
