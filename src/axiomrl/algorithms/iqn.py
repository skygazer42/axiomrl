import copy
from typing import Any

import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_iqn_network import MLPIQNetwork


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


def iqn_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    per_sample = _quantile_huber_loss(
        batch["pred_quantiles"],
        batch["target_quantiles"],
        taus=batch["taus"],
        kappa=float(batch.get("kappa", 1.0)),
    )
    loss = per_sample.mean()
    return {"loss": float(loss.detach().cpu().item())}


class IQN:
    def __init__(
        self,
        *,
        q_network: MLPIQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        num_quantiles: int,
        kappa: float = 1.0,
    ) -> None:
        if int(num_quantiles) != int(q_network.num_quantiles):
            raise ValueError(f"expected num_quantiles={q_network.num_quantiles}, got {num_quantiles}")

        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.gamma = float(gamma)
        self.target_update_interval = int(target_update_interval)
        self.num_quantiles = int(num_quantiles)
        self.kappa = float(kappa)
        self.last_td_errors: torch.Tensor | None = None

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        quantiles, taus = self.q_network(obs, num_quantiles=self.num_quantiles)
        batch_size = int(actions.shape[0])
        chosen_quantiles = quantiles.gather(
            1,
            actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles),
        ).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network.q_values(next_obs, num_quantiles=self.num_quantiles).argmax(dim=-1)
            next_target_quantiles_all, _ = self.target_network(next_obs, num_quantiles=self.num_quantiles)
            next_target_quantiles = next_target_quantiles_all.gather(
                1,
                next_actions.view(batch_size, 1, 1).expand(batch_size, 1, self.num_quantiles),
            ).squeeze(1)
            target_quantiles = rewards.unsqueeze(1) + self.gamma * (1.0 - dones).unsqueeze(1) * next_target_quantiles

        per_sample_losses = _quantile_huber_loss(
            chosen_quantiles,
            target_quantiles,
            taus=taus,
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

        q_values = chosen_quantiles.mean(dim=-1)
        target_values = target_quantiles.mean(dim=-1)
        td_errors = target_values - q_values
        self.last_td_errors = td_errors.detach().abs()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
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
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def set_train_mode(self) -> None:
        self.q_network.train(True)
        self.target_network.train(False)

    def set_eval_mode(self) -> None:
        self.q_network.eval()
        self.target_network.eval()
