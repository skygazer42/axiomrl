from __future__ import annotations

from typing import Any

import torch

from rl_training.algorithms.base import UpdateResult
from rl_training.algorithms.r2d2 import R2D2
from rl_training.models.recurrent import LSTMQNetwork
from rl_training.models.rnd import RNDModel


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    normalizer = mask.sum().clamp_min(1.0)
    return masked.sum() / normalizer


class Agent57(R2D2):
    def __init__(
        self,
        *,
        q_network: LSTMQNetwork,
        rnd_model: RNDModel,
        learning_rate: float,
        rnd_learning_rate: float,
        gamma: float,
        target_update_interval: int,
        double_q: bool = True,
        priority_eta: float = 0.9,
        intrinsic_reward_coef: float = 0.0,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=double_q,
            priority_eta=priority_eta,
        )
        self.rnd_model = rnd_model
        self.rnd_optimizer = torch.optim.Adam(self.rnd_model.predictor.parameters(), lr=rnd_learning_rate)
        self.intrinsic_reward_coef = float(intrinsic_reward_coef)

    @torch.no_grad()
    def intrinsic_reward(self, obs: object) -> torch.Tensor:
        return self.rnd_model.intrinsic_reward(obs)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        q_update = super().update(batch, global_step=global_step)
        device = next(self.q_network.parameters()).device
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)

        prediction_error = self.rnd_model.prediction_error(next_obs)
        rnd_loss = _masked_mean(prediction_error, mask)

        self.rnd_optimizer.zero_grad(set_to_none=True)
        rnd_loss.backward()
        self.rnd_optimizer.step()

        metrics = dict(q_update.metrics)
        metrics["rnd_loss"] = float(rnd_loss.detach().cpu().item())
        metrics["rnd_reward_mean"] = float(_masked_mean(prediction_error.detach(), mask).cpu().item())
        metrics["intrinsic_reward_coef"] = self.intrinsic_reward_coef
        return UpdateResult(metrics=metrics, num_gradient_steps=q_update.num_gradient_steps + 1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "rnd_model": self.rnd_model.state_dict(),
                "rnd_optimizer": self.rnd_optimizer.state_dict(),
                "intrinsic_reward_coef": self.intrinsic_reward_coef,
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.rnd_model.load_state_dict(state_dict["rnd_model"])
        self.rnd_optimizer.load_state_dict(state_dict["rnd_optimizer"])
        self.intrinsic_reward_coef = float(state_dict.get("intrinsic_reward_coef", self.intrinsic_reward_coef))

    def set_train_mode(self) -> None:
        super().set_train_mode()
        self.rnd_model.train(True)

    def set_eval_mode(self) -> None:
        super().set_eval_mode()
        self.rnd_model.eval()
