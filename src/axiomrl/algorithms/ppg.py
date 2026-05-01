import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from axiomrl.algorithms._advantage_utils import normalize_advantages
from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.cnn import CNNPPGModel
from axiomrl.models.mlp_ppg import MLPPPGModel


def _ppg_policy_loss_terms(
    minibatch: dict[str, torch.Tensor],
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, torch.Tensor]:
    old_logprobs = minibatch.get("old_logprobs", minibatch["logprobs"])
    advantages = normalize_advantages(minibatch["advantages"])
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


def ppg_loss(
    minibatch: dict[str, torch.Tensor],
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, float]:
    terms = _ppg_policy_loss_terms(
        minibatch,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _ppg_auxiliary_loss_terms(
    batch: dict[str, torch.Tensor],
    *,
    aux_value_coef: float,
    behavior_clone_coef: float,
    value_clone_coef: float,
) -> dict[str, torch.Tensor]:
    teacher_probs = torch.softmax(batch["teacher_logits"], dim=-1)
    student_log_probs = torch.log_softmax(batch["policy_logits"], dim=-1)
    behavior_clone_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    aux_value_loss = 0.5 * F.mse_loss(batch["auxiliary_values"], batch["returns"])
    value_clone_loss = 0.5 * F.mse_loss(batch["values"], batch["auxiliary_values"].detach())
    auxiliary_loss = (
        aux_value_coef * aux_value_loss
        + behavior_clone_coef * behavior_clone_loss
        + value_clone_coef * value_clone_loss
    )
    return {
        "auxiliary_loss": auxiliary_loss,
        "aux_value_loss": aux_value_loss,
        "behavior_clone_loss": behavior_clone_loss,
        "value_clone_loss": value_clone_loss,
    }


def ppg_auxiliary_loss(
    batch: dict[str, torch.Tensor],
    *,
    aux_value_coef: float,
    behavior_clone_coef: float,
    value_clone_coef: float,
) -> dict[str, float]:
    terms = _ppg_auxiliary_loss_terms(
        batch,
        aux_value_coef=aux_value_coef,
        behavior_clone_coef=behavior_clone_coef,
        value_clone_coef=value_clone_coef,
    )
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


class PPG:
    def __init__(
        self,
        *,
        model: MLPPPGModel | CNNPPGModel,
        learning_rate: float,
        aux_learning_rate: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        aux_value_coef: float,
        behavior_clone_coef: float,
        value_clone_coef: float,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.model = model
        self.policy = model
        self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
        self.auxiliary_optimizer = torch.optim.Adam(self.model.parameters(), lr=aux_learning_rate, weight_decay=0.0)
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.aux_value_coef = aux_value_coef
        self.behavior_clone_coef = behavior_clone_coef
        self.value_clone_coef = value_clone_coef
        self.max_grad_norm = max_grad_norm
        self.policy_update_count = 0
        self.auxiliary_update_count = 0

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        evaluated = self.model.evaluate_actions(obs, actions)

        minibatch = {
            "logprobs": torch.as_tensor(batch["logprobs"], dtype=torch.float32),
            "advantages": torch.as_tensor(batch["advantages"], dtype=torch.float32),
            "returns": torch.as_tensor(batch["returns"], dtype=torch.float32),
            "new_logprobs": evaluated["logprobs"],
            "new_values": evaluated["values"],
            "entropy": evaluated["entropy"],
        }

        terms = _ppg_policy_loss_terms(
            minibatch,
            clip_coef=self.clip_coef,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
        )

        self.policy_optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.policy_optimizer.step()
        self.policy_update_count += 1

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        metrics["policy_updates"] = float(self.policy_update_count)
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def snapshot_teacher_model(self) -> MLPPPGModel | CNNPPGModel:
        teacher_model = copy.deepcopy(self.model)
        teacher_model.eval()
        return teacher_model

    def auxiliary_update(
        self,
        batch: dict[str, Any],
        *,
        teacher_model: MLPPPGModel | CNNPPGModel,
        global_step: int,
    ) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)

        with torch.no_grad():
            teacher_logits = teacher_model.policy_logits(obs)

        terms = _ppg_auxiliary_loss_terms(
            {
                "policy_logits": self.model.policy_logits(obs),
                "teacher_logits": teacher_logits,
                "values": self.model.values(obs),
                "auxiliary_values": self.model.auxiliary_values(obs),
                "returns": returns,
            },
            aux_value_coef=self.aux_value_coef,
            behavior_clone_coef=self.behavior_clone_coef,
            value_clone_coef=self.value_clone_coef,
        )

        self.auxiliary_optimizer.zero_grad(set_to_none=True)
        terms["auxiliary_loss"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.auxiliary_optimizer.step()
        self.auxiliary_update_count += 1

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        metrics["auxiliary_updates"] = float(self.auxiliary_update_count)
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "auxiliary_optimizer": self.auxiliary_optimizer.state_dict(),
            "policy_update_count": self.policy_update_count,
            "auxiliary_update_count": self.auxiliary_update_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.auxiliary_optimizer.load_state_dict(state_dict["auxiliary_optimizer"])
        self.policy_update_count = int(state_dict.get("policy_update_count", 0))
        self.auxiliary_update_count = int(state_dict.get("auxiliary_update_count", 0))

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()
