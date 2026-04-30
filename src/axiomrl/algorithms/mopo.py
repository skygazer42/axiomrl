from __future__ import annotations

from typing import Any

import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.sac import SAC
from axiomrl.models.mlp_mopo import MLPMOPOEnsembleModel
from axiomrl.models.mlp_sac import MLPSACModel


def _mopo_model_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = batch["targets"].unsqueeze(0).expand_as(batch["predicted_means"])
    inv_vars = torch.exp(-batch["predicted_logvars"])
    nll = 0.5 * ((targets - batch["predicted_means"]).pow(2) * inv_vars + batch["predicted_logvars"])
    reward_mae = (batch["predicted_means"][..., -1] - targets[..., -1]).abs().mean()
    delta_obs_mae = (batch["predicted_means"][..., :-1] - targets[..., :-1]).abs().mean()
    return {
        "mopo_model_loss": nll.mean(),
        "reward_mae": reward_mae,
        "delta_obs_mae": delta_obs_mae,
        "ensemble_disagreement": batch["ensemble_disagreement"].mean(),
    }


def mopo_model_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    tensor_batch = {
        "predicted_means": torch.as_tensor(batch["predicted_means"], dtype=torch.float32),
        "predicted_logvars": torch.as_tensor(batch["predicted_logvars"], dtype=torch.float32),
        "targets": torch.as_tensor(batch["targets"], dtype=torch.float32),
        "ensemble_disagreement": torch.as_tensor(batch["ensemble_disagreement"], dtype=torch.float32),
    }
    terms = _mopo_model_loss_terms(tensor_batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class MOPO:
    def __init__(
        self,
        *,
        policy_model: MLPSACModel,
        dynamics_model: MLPMOPOEnsembleModel,
        policy_learning_rate: float,
        model_learning_rate: float,
        gamma: float,
        alpha: float,
        tau: float,
        penalty_coef: float,
    ) -> None:
        if float(penalty_coef) < 0.0:
            raise ValueError(f"penalty_coef must be >= 0, got {penalty_coef}")

        self.model = policy_model
        self.policy_model = policy_model
        self.policy = policy_model
        self.dynamics_model = dynamics_model
        self.policy_algorithm = SAC(
            model=self.policy_model,
            learning_rate=float(policy_learning_rate),
            gamma=float(gamma),
            alpha=float(alpha),
            tau=float(tau),
        )
        self.model_optimizer = torch.optim.Adam(
            self.dynamics_model.ensemble_parameters(), lr=float(model_learning_rate), weight_decay=0.0
        )
        self.penalty_coef = float(penalty_coef)

    def update_model(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=obs.device)

        predicted_means, predicted_logvars = self.dynamics_model.predict_distribution(obs, actions)
        targets = torch.cat([next_obs - obs, rewards.unsqueeze(-1)], dim=-1)
        disagreement = predicted_means.std(dim=0).mean(dim=-1)
        terms = _mopo_model_loss_terms(
            {
                "predicted_means": predicted_means,
                "predicted_logvars": predicted_logvars,
                "targets": targets,
                "ensemble_disagreement": disagreement,
            }
        )

        self.model_optimizer.zero_grad(set_to_none=True)
        terms["mopo_model_loss"].backward()
        self.model_optimizer.step()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def sample_synthetic_transition(self, obs: object, actions: object) -> dict[str, torch.Tensor]:
        sample = self.dynamics_model.sample_transition(obs, actions)
        penalized_rewards = sample["rewards"] - self.penalty_coef * sample["disagreement"]
        return {
            "next_obs": sample["next_obs"],
            "rewards": penalized_rewards,
            "raw_rewards": sample["rewards"],
            "disagreement": sample["disagreement"],
        }

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        return self.policy_algorithm.update(batch, global_step=global_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_algorithm": self.policy_algorithm.state_dict(),
            "dynamics_model": self.dynamics_model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "penalty_coef": self.penalty_coef,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_algorithm.load_state_dict(state_dict["policy_algorithm"])
        self.dynamics_model.load_state_dict(state_dict["dynamics_model"])
        self.model_optimizer.load_state_dict(state_dict["model_optimizer"])
        self.penalty_coef = float(state_dict.get("penalty_coef", self.penalty_coef))

    def set_train_mode(self) -> None:
        self.policy_algorithm.set_train_mode()
        self.dynamics_model.train(True)

    def set_eval_mode(self) -> None:
        self.policy_algorithm.set_eval_mode()
        self.dynamics_model.eval()
