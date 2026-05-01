from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.algorithms.muzero import MuZero, MuZeroMCTSConfig
from axiomrl.models.muzero import MuZeroModel


class EfficientZero(MuZero):
    def __init__(
        self,
        *,
        model: MuZeroModel,
        learning_rate: float,
        gamma: float,
        mcts_config: MuZeroMCTSConfig,
        unroll_steps: int,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        consistency_loss_weight: float = 1.0,
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
        self.consistency_loss_weight = float(consistency_loss_weight)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        device = next(self.model.parameters()).device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        target_obs = torch.as_tensor(batch["target_obs"], dtype=torch.float32, device=device)
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
            bootstrap_value = self.model.initial_inference(bootstrap_obs).value
            bootstrap_value = bootstrap_value * state_valid[:, -1]

            value_targets = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32, device=device)
            value_targets[:, -1] = bootstrap_value
            for step in range(unroll_steps - 1, -1, -1):
                value_targets[:, step] = rewards[:, step] + self.gamma * value_targets[:, step + 1] * (
                    1.0 - dones[:, step]
                )

            flat_target_obs = target_obs.reshape(batch_size * unroll_steps, *target_obs.shape[2:])
            target_hidden = self.model.initial_inference(flat_target_obs).hidden_state.reshape(
                batch_size, unroll_steps, -1
            )

        initial = self.model.initial_inference(obs)
        pred_policy_logits = [initial.policy_logits]
        pred_values = [initial.value]
        pred_rewards: list[torch.Tensor] = []
        pred_hidden_states: list[torch.Tensor] = []

        hidden = initial.hidden_state
        for step in range(unroll_steps):
            recurrent = self.model.recurrent_inference(hidden, actions[:, step])
            hidden = recurrent.hidden_state
            pred_hidden_states.append(hidden)
            pred_rewards.append(recurrent.reward)
            pred_policy_logits.append(recurrent.policy_logits)
            pred_values.append(recurrent.value)

        pred_policy = torch.stack(pred_policy_logits, dim=1)
        pred_value = torch.stack(pred_values, dim=1)
        pred_reward = torch.stack(pred_rewards, dim=1) if pred_rewards else torch.zeros_like(rewards)
        pred_hidden = torch.stack(pred_hidden_states, dim=1) if pred_hidden_states else torch.zeros_like(target_hidden)

        log_probs = F.log_softmax(pred_policy, dim=-1)
        policy_ce = -(target_policies * log_probs).sum(dim=-1)
        policy_loss = (policy_ce * state_valid).sum() / (state_valid.sum() + 1e-8)

        value_mse = 0.5 * (pred_value - value_targets).pow(2)
        value_loss = (value_mse * state_valid).sum() / (state_valid.sum() + 1e-8)

        transition_valid = state_valid[:, :-1]
        reward_mse = 0.5 * (pred_reward - rewards).pow(2)
        reward_loss = (reward_mse * transition_valid).sum() / (transition_valid.sum() + 1e-8)

        consistency_mse = 0.5 * (pred_hidden - target_hidden).pow(2).mean(dim=-1)
        consistency_loss = (consistency_mse * transition_valid).sum() / (transition_valid.sum() + 1e-8)

        loss = (
            self.policy_loss_weight * policy_loss
            + self.value_loss_weight * value_loss
            + self.reward_loss_weight * reward_loss
            + self.consistency_loss_weight * consistency_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "reward_loss": float(reward_loss.detach().cpu().item()),
            "consistency_loss": float(consistency_loss.detach().cpu().item()),
            "value_mean": float(pred_value[:, 0].mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["consistency_loss_weight"] = self.consistency_loss_weight
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.consistency_loss_weight = float(state_dict.get("consistency_loss_weight", self.consistency_loss_weight))
