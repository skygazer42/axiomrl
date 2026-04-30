from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.muzero import MuZero, MuZeroMCTSConfig
from axiomrl.models.scalezero import ScaleZeroModel


class ScaleZero(MuZero):
    def __init__(
        self,
        *,
        model: ScaleZeroModel,
        learning_rate: float,
        gamma: float,
        mcts_config: MuZeroMCTSConfig,
        unroll_steps: int,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        max_grad_norm: float = 10.0,
    ) -> None:
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            mcts_config=mcts_config,
            unroll_steps=unroll_steps,
            value_loss_weight=value_loss_weight,
            reward_loss_weight=reward_loss_weight,
            policy_loss_weight=policy_loss_weight,
            max_grad_norm=max_grad_norm,
        )

    @property
    def scalezero_model(self) -> ScaleZeroModel:
        return self.model

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        device = next(self.model.parameters()).device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
        target_policies = torch.as_tensor(batch["target_policies"], dtype=torch.float32, device=device)
        bootstrap_obs = torch.as_tensor(batch["bootstrap_obs"], dtype=torch.float32, device=device)

        batch_size = int(obs.shape[0])
        unroll_steps = int(actions.shape[1])

        state_valid = torch.ones((batch_size, unroll_steps + 1), dtype=torch.float32, device=device)
        if unroll_steps > 0:
            for step in range(1, unroll_steps + 1):
                state_valid[:, step] = state_valid[:, step - 1] * (1.0 - dones[:, step - 1])

        with torch.no_grad():
            bootstrap_value = self.scalezero_model.initial_inference(bootstrap_obs).value
            bootstrap_value = bootstrap_value * state_valid[:, -1]

            value_targets = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32, device=device)
            value_targets[:, -1] = bootstrap_value
            for step in range(unroll_steps - 1, -1, -1):
                value_targets[:, step] = rewards[:, step] + self.gamma * value_targets[:, step + 1] * (
                    1.0 - dones[:, step]
                )

        initial, initial_info = self.scalezero_model.initial_inference_with_info(obs)
        pred_policy_logits = [initial.policy_logits]
        pred_values = [initial.value]
        pred_rewards: list[torch.Tensor] = []
        prediction_gate_entropies = [self.scalezero_model.gate_entropy(initial_info["prediction_gate_probs"]).mean()]
        dynamics_gate_entropies: list[torch.Tensor] = []

        hidden = initial.hidden_state
        for step in range(unroll_steps):
            recurrent, info = self.scalezero_model.recurrent_inference_with_info(hidden, actions[:, step])
            hidden = recurrent.hidden_state
            pred_rewards.append(recurrent.reward)
            pred_policy_logits.append(recurrent.policy_logits)
            pred_values.append(recurrent.value)
            dynamics_gate_entropies.append(self.scalezero_model.gate_entropy(info["dynamics_gate_probs"]).mean())
            prediction_gate_entropies.append(self.scalezero_model.gate_entropy(info["prediction_gate_probs"]).mean())

        pred_policy = torch.stack(pred_policy_logits, dim=1)
        pred_value = torch.stack(pred_values, dim=1)
        pred_reward = torch.stack(pred_rewards, dim=1) if pred_rewards else torch.zeros_like(rewards)

        log_probs = F.log_softmax(pred_policy, dim=-1)
        policy_ce = -(target_policies * log_probs).sum(dim=-1)
        policy_loss = (policy_ce * state_valid).sum() / (state_valid.sum() + 1e-8)

        value_mse = 0.5 * (pred_value - value_targets).pow(2)
        value_loss = (value_mse * state_valid).sum() / (state_valid.sum() + 1e-8)

        transition_valid = state_valid[:, :-1]
        reward_mse = 0.5 * (pred_reward - rewards).pow(2)
        reward_loss = (reward_mse * transition_valid).sum() / (transition_valid.sum() + 1e-8)

        loss = (
            self.policy_loss_weight * policy_loss
            + self.value_loss_weight * value_loss
            + self.reward_loss_weight * reward_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        all_gate_entropies = prediction_gate_entropies + dynamics_gate_entropies
        metrics = {
            "scalezero_loss": float(loss.detach().cpu().item()),
            "scalezero_policy_loss": float(policy_loss.detach().cpu().item()),
            "scalezero_value_loss": float(value_loss.detach().cpu().item()),
            "scalezero_reward_loss": float(reward_loss.detach().cpu().item()),
            "scalezero_prediction_moe_entropy": float(
                torch.stack(prediction_gate_entropies).mean().detach().cpu().item()
            ),
            "scalezero_dynamics_moe_entropy": float(torch.stack(dynamics_gate_entropies).mean().detach().cpu().item())
            if dynamics_gate_entropies
            else 0.0,
            "scalezero_moe_entropy": float(torch.stack(all_gate_entropies).mean().detach().cpu().item()),
            "scalezero_num_experts": float(self.scalezero_model.num_experts),
            "scalezero_value_mean": float(pred_value[:, 0].mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)
