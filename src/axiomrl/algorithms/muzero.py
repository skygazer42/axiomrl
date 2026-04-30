from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.muzero import MuZeroModel
from axiomrl.policies.base import PolicyOutput


@dataclass(frozen=True, slots=True)
class MuZeroMCTSConfig:
    num_simulations: int = 25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.pb_c_base <= 0:
            raise ValueError(f"pb_c_base must be > 0, got {self.pb_c_base}")
        if self.pb_c_init <= 0:
            raise ValueError(f"pb_c_init must be > 0, got {self.pb_c_init}")
        if self.root_dirichlet_alpha <= 0:
            raise ValueError(f"root_dirichlet_alpha must be > 0, got {self.root_dirichlet_alpha}")
        if not 0.0 <= self.root_exploration_fraction <= 1.0:
            raise ValueError(f"root_exploration_fraction must be in [0, 1], got {self.root_exploration_fraction}")


@dataclass(slots=True)
class _Node:
    prior: float
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, _Node] = field(default_factory=dict)
    hidden_state: torch.Tensor | None = None
    expanded: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


def _puct_score(
    parent: _Node,
    child: _Node,
    *,
    pb_c_base: float,
    pb_c_init: float,
) -> float:
    pb_c = math.log((parent.visit_count + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1.0)
    prior_score = pb_c * child.prior
    return child.value() + prior_score


def _expand_node(
    node: _Node,
    *,
    action_dim: int,
    priors: np.ndarray,
) -> None:
    node.children = {action: _Node(prior=float(priors[action])) for action in range(int(action_dim))}
    node.expanded = True


def _apply_root_dirichlet_noise(
    priors: np.ndarray,
    *,
    alpha: float,
    fraction: float,
) -> np.ndarray:
    if fraction <= 0.0:
        return priors
    noise = np.random.dirichlet([float(alpha)] * int(priors.shape[0])).astype(np.float32)
    return (1.0 - float(fraction)) * priors + float(fraction) * noise


def run_muzero_mcts(
    *,
    model: MuZeroModel,
    obs: object,
    mcts: MuZeroMCTSConfig,
    gamma: float,
    add_root_noise: bool,
) -> tuple[np.ndarray, float]:
    model.eval()
    with torch.no_grad():
        root_out = model.initial_inference(
            torch.as_tensor(obs, dtype=torch.float32, device=next(model.parameters()).device)
        )

    action_dim = int(model.action_dim)
    root = _Node(prior=1.0, reward=0.0, visit_count=0, value_sum=0.0, hidden_state=root_out.hidden_state)
    root_priors = torch.softmax(root_out.policy_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if add_root_noise:
        root_priors = _apply_root_dirichlet_noise(
            root_priors,
            alpha=mcts.root_dirichlet_alpha,
            fraction=mcts.root_exploration_fraction,
        )
    _expand_node(root, action_dim=action_dim, priors=root_priors)

    for _ in range(int(mcts.num_simulations)):
        node = root
        search_path = [node]
        action_path: list[int] = []

        while node.expanded and node.children:
            best_action = max(
                node.children.items(),
                key=lambda item: _puct_score(
                    node,
                    item[1],
                    pb_c_base=mcts.pb_c_base,
                    pb_c_init=mcts.pb_c_init,
                ),
            )[0]
            action_path.append(best_action)
            child = node.children[best_action]
            node = child
            search_path.append(node)

            if node.hidden_state is None:
                parent_hidden = search_path[-2].hidden_state
                if parent_hidden is None:
                    raise RuntimeError("MCTS encountered missing parent hidden state")
                with torch.no_grad():
                    recurrent = model.recurrent_inference(
                        parent_hidden, torch.tensor([best_action], device=parent_hidden.device)
                    )
                node.hidden_state = recurrent.hidden_state
                node.reward = float(recurrent.reward.squeeze(0).detach().cpu().item())

                priors = (
                    torch.softmax(recurrent.policy_logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                )
                _expand_node(node, action_dim=action_dim, priors=priors)

                leaf_value = float(recurrent.value.squeeze(0).detach().cpu().item())
                break
        else:
            # If we somehow reach an unexpanded node (should not happen for root), treat its value as 0.
            leaf_value = 0.0

        value = leaf_value
        for backed in reversed(search_path):
            backed.visit_count += 1
            backed.value_sum += value
            value = backed.reward + float(gamma) * value

    visit_counts = np.array([root.children[a].visit_count for a in range(action_dim)], dtype=np.float32)
    total_visits = float(visit_counts.sum())
    if total_visits <= 0:
        probs = np.full((action_dim,), 1.0 / float(action_dim), dtype=np.float32)
    else:
        probs = visit_counts / total_visits
    return probs, root.value()


def _select_action_from_probs(probs: np.ndarray, *, temperature: float, deterministic: bool) -> int:
    action_dim = int(probs.shape[0])
    if deterministic or temperature <= 1e-8:
        return int(probs.argmax())

    adjusted = probs ** (1.0 / float(temperature))
    adjusted_sum = float(adjusted.sum())
    if adjusted_sum <= 0:
        adjusted = np.full((action_dim,), 1.0 / float(action_dim), dtype=np.float32)
    else:
        adjusted = adjusted / adjusted_sum
    return int(np.random.choice(np.arange(action_dim), p=adjusted))


class MuZero:
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
        max_grad_norm: float = 10.0,
    ) -> None:
        if unroll_steps < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")

        self.model = model
        self.policy = self
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), weight_decay=0.0)
        self.gamma = float(gamma)
        self.mcts_config = mcts_config
        self.unroll_steps = int(unroll_steps)
        self.value_loss_weight = float(value_loss_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.policy_loss_weight = float(policy_loss_weight)
        self.max_grad_norm = float(max_grad_norm)

    def plan(
        self,
        obs: object,
        *,
        temperature: float,
        add_root_noise: bool,
        deterministic: bool,
        root_exploration_fraction: float | None = None,
        num_simulations: int | None = None,
    ) -> tuple[int, np.ndarray, float]:
        mcts_config = self.mcts_config
        if root_exploration_fraction is not None or num_simulations is not None:
            mcts_config = replace(
                mcts_config,
                root_exploration_fraction=(
                    float(root_exploration_fraction)
                    if root_exploration_fraction is not None
                    else mcts_config.root_exploration_fraction
                ),
                num_simulations=int(num_simulations) if num_simulations is not None else mcts_config.num_simulations,
            )
        probs, root_value = run_muzero_mcts(
            model=self.model,
            obs=obs,
            mcts=mcts_config,
            gamma=self.gamma,
            add_root_noise=add_root_noise,
        )
        action = _select_action_from_probs(probs, temperature=temperature, deterministic=deterministic)
        return action, probs.astype(np.float32), float(root_value)

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        del state
        action, _, _ = self.plan(
            obs,
            temperature=0.0 if deterministic else 1.0,
            add_root_noise=False,
            deterministic=deterministic,
        )
        return PolicyOutput(
            actions=torch.tensor([action], dtype=torch.int64), logprobs=None, values=None, entropy=None, state=None
        )

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        del global_step
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=next(self.model.parameters()).device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=obs.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=obs.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=obs.device)
        target_policies = torch.as_tensor(batch["target_policies"], dtype=torch.float32, device=obs.device)
        bootstrap_obs = torch.as_tensor(batch["bootstrap_obs"], dtype=torch.float32, device=obs.device)

        batch_size = int(obs.shape[0])
        unroll_steps = int(actions.shape[1])

        # Valid state mask: state at t+k is invalid if done at any previous transition.
        state_valid = torch.ones((batch_size, unroll_steps + 1), dtype=torch.float32, device=obs.device)
        if unroll_steps > 0:
            for k in range(1, unroll_steps + 1):
                state_valid[:, k] = state_valid[:, k - 1] * (1.0 - dones[:, k - 1])

        with torch.no_grad():
            bootstrap_value = self.model.initial_inference(bootstrap_obs).value
            bootstrap_value = bootstrap_value * state_valid[:, -1]

            value_targets = torch.zeros((batch_size, unroll_steps + 1), dtype=torch.float32, device=obs.device)
            value_targets[:, -1] = bootstrap_value
            for t in range(unroll_steps - 1, -1, -1):
                value_targets[:, t] = rewards[:, t] + self.gamma * value_targets[:, t + 1] * (1.0 - dones[:, t])

        # Unroll predictions.
        initial = self.model.initial_inference(obs)
        pred_policy_logits = [initial.policy_logits]
        pred_values = [initial.value]
        pred_rewards: list[torch.Tensor] = []

        hidden = initial.hidden_state
        for t in range(unroll_steps):
            recurrent = self.model.recurrent_inference(hidden, actions[:, t])
            hidden = recurrent.hidden_state
            pred_rewards.append(recurrent.reward)
            pred_policy_logits.append(recurrent.policy_logits)
            pred_values.append(recurrent.value)

        pred_policy = torch.stack(pred_policy_logits, dim=1)  # (B, T+1, A)
        pred_value = torch.stack(pred_values, dim=1)  # (B, T+1)
        pred_reward = torch.stack(pred_rewards, dim=1) if pred_rewards else torch.zeros_like(rewards)  # (B, T)

        # Losses.
        log_probs = F.log_softmax(pred_policy, dim=-1)
        policy_ce = -(target_policies * log_probs).sum(dim=-1)  # (B, T+1)
        policy_loss = (policy_ce * state_valid).sum() / (state_valid.sum() + 1e-8)

        value_mse = 0.5 * (pred_value - value_targets).pow(2)  # (B, T+1)
        value_loss = (value_mse * state_valid).sum() / (state_valid.sum() + 1e-8)

        transition_valid = state_valid[:, :-1]
        reward_mse = 0.5 * (pred_reward - rewards).pow(2)  # (B, T)
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

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "reward_loss": float(reward_loss.detach().cpu().item()),
            "value_mean": float(pred_value[:, 0].mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gamma": self.gamma,
            "unroll_steps": self.unroll_steps,
            "value_loss_weight": self.value_loss_weight,
            "reward_loss_weight": self.reward_loss_weight,
            "policy_loss_weight": self.policy_loss_weight,
            "max_grad_norm": self.max_grad_norm,
            "mcts_config": {
                "num_simulations": self.mcts_config.num_simulations,
                "pb_c_base": self.mcts_config.pb_c_base,
                "pb_c_init": self.mcts_config.pb_c_init,
                "root_dirichlet_alpha": self.mcts_config.root_dirichlet_alpha,
                "root_exploration_fraction": self.mcts_config.root_exploration_fraction,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.gamma = float(state_dict.get("gamma", self.gamma))
        self.unroll_steps = int(state_dict.get("unroll_steps", self.unroll_steps))
        self.value_loss_weight = float(state_dict.get("value_loss_weight", self.value_loss_weight))
        self.reward_loss_weight = float(state_dict.get("reward_loss_weight", self.reward_loss_weight))
        self.policy_loss_weight = float(state_dict.get("policy_loss_weight", self.policy_loss_weight))
        self.max_grad_norm = float(state_dict.get("max_grad_norm", self.max_grad_norm))
        mcts_payload = dict(state_dict.get("mcts_config", {}))
        if mcts_payload:
            self.mcts_config = MuZeroMCTSConfig(
                num_simulations=int(mcts_payload.get("num_simulations", self.mcts_config.num_simulations)),
                pb_c_base=float(mcts_payload.get("pb_c_base", self.mcts_config.pb_c_base)),
                pb_c_init=float(mcts_payload.get("pb_c_init", self.mcts_config.pb_c_init)),
                root_dirichlet_alpha=float(
                    mcts_payload.get("root_dirichlet_alpha", self.mcts_config.root_dirichlet_alpha)
                ),
                root_exploration_fraction=float(
                    mcts_payload.get("root_exploration_fraction", self.mcts_config.root_exploration_fraction)
                ),
            )

    def set_train_mode(self) -> None:
        self.model.train(True)

    def set_eval_mode(self) -> None:
        self.model.eval()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()
