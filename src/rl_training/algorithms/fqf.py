from __future__ import annotations

import copy
from typing import Any

import torch

from rl_training.algorithms.base import UpdateResult
from rl_training.models.mlp_fqf_network import MLPFQFNetwork


def _quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    *,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    if pred_quantiles.ndim != 2 or target_quantiles.ndim != 2:
        raise ValueError("expected pred_quantiles and target_quantiles to have shape (batch, num_quantiles)")
    if taus.ndim != 2:
        raise ValueError("expected taus to have shape (batch, num_quantiles)")

    td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_errors = td_errors.abs()

    huber = torch.where(
        abs_errors <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (abs_errors - 0.5 * kappa),
    )

    indicator = (td_errors.detach() < 0).to(dtype=pred_quantiles.dtype)
    tau = taus.unsqueeze(2).to(device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    quantile_weights = (tau - indicator).abs()
    return (quantile_weights * huber).mean(dim=(1, 2))


def _fqf_fraction_loss(
    *,
    quantile_hats: torch.Tensor,
    quantiles_tau: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    if quantile_hats.ndim != 2 or quantiles_tau.ndim != 2 or taus.ndim != 2:
        raise ValueError("expected quantile tensors to have shape (batch, num_quantiles) and taus (batch, num_quantiles+1)")
    if int(quantiles_tau.shape[1]) + 2 != int(taus.shape[1]):
        raise ValueError("quantiles_tau must cover interior taus only")

    # Approximate ∂W1 / ∂τ as in common FQF implementations.
    # Use detached quantile values so the fraction proposal updates do not
    # backprop into the quantile network.
    q_hats = quantile_hats.detach()
    q_tau = quantiles_tau.detach()

    left = q_tau - q_hats[:, :-1]
    left_sign = q_tau > torch.cat([q_hats[:, :1], q_tau[:, :-1]], dim=1)
    right = q_tau - q_hats[:, 1:]
    right_sign = q_tau < torch.cat([q_tau[:, 1:], q_hats[:, -1:]], dim=1)
    gradient = torch.where(left_sign, left, -left) + torch.where(right_sign, right, -right)

    # Stop-gradient on gradient; train fractions by supplying it as the
    # effective gradient for the interior taus.
    fraction_loss = (gradient.detach() * taus[:, 1:-1]).sum(dim=1).mean()
    return fraction_loss


class FQF:
    def __init__(
        self,
        *,
        q_network: MLPFQFNetwork,
        learning_rate: float,
        fraction_learning_rate: float,
        gamma: float,
        target_update_interval: int,
        num_quantiles: int,
        kappa: float = 1.0,
        entropy_coef: float = 1e-3,
    ) -> None:
        if int(num_quantiles) != int(q_network.num_quantiles):
            raise ValueError(f"expected num_quantiles={q_network.num_quantiles}, got {num_quantiles}")
        if float(entropy_coef) < 0.0:
            raise ValueError(f"entropy_coef must be >= 0, got {entropy_coef}")

        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)

        self.quantile_optimizer = torch.optim.Adam(
            self.q_network.quantile_parameters(),
            lr=float(learning_rate),
            weight_decay=0.0,
        )
        self.fraction_optimizer = torch.optim.Adam(
            self.q_network.fraction_parameters(),
            lr=float(fraction_learning_rate),
            weight_decay=0.0,
        )
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.num_quantiles = int(num_quantiles)
        self.kappa = float(kappa)
        self.entropy_coef = float(entropy_coef)
        self.last_td_errors: torch.Tensor | None = None

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        batch_size = int(actions.shape[0])

        # 1) Update the fraction proposal network.
        fraction_out = self.q_network(obs, detach_quantiles_tau=True)
        quantile_hats_all = fraction_out.quantile_hats
        quantiles_tau_all = fraction_out.quantiles_tau

        quantile_hats = quantile_hats_all.gather(
            1,
            actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles),
        ).squeeze(1)
        quantiles_tau = quantiles_tau_all.gather(
            1,
            actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles - 1),
        ).squeeze(1)

        fraction_loss = _fqf_fraction_loss(
            quantile_hats=quantile_hats,
            quantiles_tau=quantiles_tau,
            taus=fraction_out.taus,
        )
        entropy_loss = -fraction_out.entropies.mean()
        fraction_total_loss = fraction_loss + self.entropy_coef * entropy_loss

        self.fraction_optimizer.zero_grad(set_to_none=True)
        fraction_total_loss.backward()
        self.fraction_optimizer.step()

        # 2) Update the quantile value network with the latest fractions.
        out = self.q_network(obs)
        quantile_hats_all = out.quantile_hats
        tau_hats = out.tau_hats
        quantile_hats = quantile_hats_all.gather(
            1,
            actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles),
        ).squeeze(1)

        with torch.no_grad():
            next_out_online = self.q_network(next_obs)
            next_q_values = (next_out_online.quantile_hats * (next_out_online.taus[:, 1:] - next_out_online.taus[:, :-1]).unsqueeze(1)).sum(
                dim=-1
            )
            next_actions = next_q_values.argmax(dim=-1)
            next_target_quantiles_all = self.target_network.quantiles(next_obs, next_out_online.tau_hats)
            next_target_quantiles = next_target_quantiles_all.gather(
                1,
                next_actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles),
            ).squeeze(1)
            target_quantiles = rewards.unsqueeze(1) + self.gamma * (1.0 - dones).unsqueeze(1) * next_target_quantiles

        per_sample_losses = _quantile_huber_loss(
            quantile_hats,
            target_quantiles,
            taus=tau_hats,
            kappa=self.kappa,
        )
        weights = batch.get("weights")
        if weights is None:
            loss = per_sample_losses.mean()
        else:
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=per_sample_losses.device)
            if weight_tensor.ndim != 1:
                weight_tensor = weight_tensor.reshape(-1)
            loss = (weight_tensor * per_sample_losses).mean()

        q_values = quantile_hats.mean(dim=-1)
        target_values = target_quantiles.mean(dim=-1)
        td_errors = target_values - q_values
        self.last_td_errors = td_errors.detach().abs()

        self.quantile_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.quantile_optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "fraction_loss": float(fraction_loss.detach().cpu().item()),
            "entropy_loss": float(entropy_loss.detach().cpu().item()),
            "q_value_mean": float(q_values.mean().detach().cpu().item()),
            "target_mean": float(target_values.mean().detach().cpu().item()),
            "td_error_mean": float(td_errors.abs().mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "quantile_optimizer": self.quantile_optimizer.state_dict(),
            "fraction_optimizer": self.fraction_optimizer.state_dict(),
            "entropy_coef": self.entropy_coef,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.quantile_optimizer.load_state_dict(state_dict["quantile_optimizer"])
        self.fraction_optimizer.load_state_dict(state_dict["fraction_optimizer"])
        self.entropy_coef = float(state_dict.get("entropy_coef", self.entropy_coef))

    def set_train_mode(self) -> None:
        self.q_network.train(True)
        self.target_network.train(False)

    def set_eval_mode(self) -> None:
        self.q_network.eval()
        self.target_network.eval()

