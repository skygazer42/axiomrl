import copy
from typing import Any

import torch
from torch.nn import functional as F

from axiomrl.algorithms.base import UpdateResult
from axiomrl.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from axiomrl.models.mlp_dueling_q_network import MLPDuelingQNetwork
from axiomrl.models.mlp_noisy_q_network import MLPNoisyQNetwork
from axiomrl.models.mlp_q_network import MLPQNetwork


def _dqn_loss_terms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    chosen_q_values = batch["q_values"].gather(1, batch["actions"].long().unsqueeze(-1)).squeeze(-1)
    td_error = batch["target_q_values"] - chosen_q_values
    loss = F.smooth_l1_loss(chosen_q_values, batch["target_q_values"])

    return {
        "loss": loss,
        "q_value_mean": chosen_q_values.mean(),
        "target_mean": batch["target_q_values"].mean(),
        "td_error_mean": td_error.abs().mean(),
    }


def dqn_loss(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    terms = _dqn_loss_terms(batch)
    return {name: float(value.detach().cpu().item()) for name, value in terms.items()}


def _epsilon_greedy_action_probs(q_values: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    action_dim = q_values.shape[-1]
    greedy_actions = q_values.argmax(dim=-1, keepdim=True)
    probs = torch.full_like(q_values, fill_value=float(epsilon) / float(action_dim))
    probs.scatter_add_(
        1, greedy_actions, torch.full_like(greedy_actions, fill_value=1.0 - float(epsilon), dtype=probs.dtype)
    )
    return probs


def _soft_action_log_probs(q_values: torch.Tensor, *, temperature: float) -> torch.Tensor:
    return torch.log_softmax(q_values / float(temperature), dim=-1)


def _soft_state_value(
    policy_q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    log_policy = _soft_action_log_probs(policy_q_values, temperature=temperature)
    policy = log_policy.exp()
    return (policy * (target_q_values - float(temperature) * log_policy)).sum(dim=-1)


def _weighted_smooth_l1_loss(
    chosen_q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_item_losses = F.smooth_l1_loss(chosen_q_values, target_q_values, reduction="none")
    if weights is None:
        return per_item_losses.mean()

    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=per_item_losses.device)
    if weight_tensor.ndim != 1:
        weight_tensor = weight_tensor.reshape(-1)
    return (weight_tensor * per_item_losses).mean()


def _mellowmax(q_values: torch.Tensor, *, omega: float) -> torch.Tensor:
    scaled = q_values * float(omega)
    max_scaled = scaled.max(dim=-1, keepdim=True).values
    return (max_scaled.squeeze(-1) + torch.log(torch.exp(scaled - max_scaled).mean(dim=-1))) / float(omega)


class DQN:
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        double_q: bool = False,
    ) -> None:
        self.q_network = q_network
        self.policy = q_network
        self.target_network = copy.deepcopy(q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0)
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.double_q = double_q
        self.last_td_errors: torch.Tensor | None = None

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        if self.double_q:
            next_actions = self.q_network(next_obs).argmax(dim=-1, keepdim=True)
            return self.target_network(next_obs).gather(1, next_actions).squeeze(-1)
        return self.target_network(next_obs).max(dim=-1).values

    def _compute_target_q_values(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        del obs, actions, q_values
        next_q_values = self._next_target_q_values(next_obs)
        return rewards + self.gamma * next_q_values * (1.0 - dones)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_network(obs)
        with torch.no_grad():
            target_q_values = self._compute_target_q_values(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                q_values=q_values,
            )

        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        td_errors = target_q_values - chosen_q_values
        self.last_td_errors = td_errors.detach().abs()
        loss = _weighted_smooth_l1_loss(chosen_q_values, target_q_values, weights=batch.get("weights"))

        terms = {
            "loss": loss,
            "q_value_mean": chosen_q_values.mean(),
            "target_mean": target_q_values.mean(),
            "td_error_mean": td_errors.abs().mean(),
        }

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
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


class DoubleDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )


class RainbowDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
    ) -> None:
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
            double_q=True,
        )


class DuelingDQN(DQN):
    pass


class NoisyDQN(DQN):
    pass


class PrioritizedDQN(DQN):
    pass


class MellowmaxDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        mellowmax_omega: float,
    ) -> None:
        if float(mellowmax_omega) <= 0.0:
            raise ValueError(f"mellowmax_omega must be > 0, got {mellowmax_omega}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.mellowmax_omega = float(mellowmax_omega)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_network(next_obs)
        return _mellowmax(next_q_values, omega=self.mellowmax_omega)


class SoftDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        entropy_temperature: float,
    ) -> None:
        if float(entropy_temperature) <= 0.0:
            raise ValueError(f"entropy_temperature must be > 0, got {entropy_temperature}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.entropy_temperature = float(entropy_temperature)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_network(next_obs)
        return self.entropy_temperature * torch.logsumexp(next_q_values / self.entropy_temperature, dim=-1)


class SoftDoubleDQN(SoftDQN):
    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        online_next_q_values = self.q_network(next_obs)
        target_next_q_values = self.target_network(next_obs)
        return _soft_state_value(
            online_next_q_values,
            target_next_q_values,
            temperature=self.entropy_temperature,
        )


class ExpectedSARSA(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        target_epsilon: float,
    ) -> None:
        if not 0.0 <= float(target_epsilon) <= 1.0:
            raise ValueError(f"target_epsilon must be in [0, 1], got {target_epsilon}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.target_epsilon = float(target_epsilon)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_network(next_obs)
        action_probs = _epsilon_greedy_action_probs(next_q_values, epsilon=self.target_epsilon)
        return (action_probs * next_q_values).sum(dim=-1)


class ExpectedDoubleDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        target_epsilon: float,
    ) -> None:
        if not 0.0 <= float(target_epsilon) <= 1.0:
            raise ValueError(f"target_epsilon must be in [0, 1], got {target_epsilon}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.target_epsilon = float(target_epsilon)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        online_next_q_values = self.q_network(next_obs)
        target_next_q_values = self.target_network(next_obs)
        action_probs = _epsilon_greedy_action_probs(online_next_q_values, epsilon=self.target_epsilon)
        return (action_probs * target_next_q_values).sum(dim=-1)


class BoltzmannDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        boltzmann_temperature: float,
    ) -> None:
        if float(boltzmann_temperature) <= 0.0:
            raise ValueError(f"boltzmann_temperature must be > 0, got {boltzmann_temperature}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.boltzmann_temperature = float(boltzmann_temperature)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_network(next_obs)
        next_log_policy = _soft_action_log_probs(next_q_values, temperature=self.boltzmann_temperature)
        next_policy = next_log_policy.exp()
        return (next_policy * next_q_values).sum(dim=-1)


class BoltzmannDoubleDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        boltzmann_temperature: float,
    ) -> None:
        if float(boltzmann_temperature) <= 0.0:
            raise ValueError(f"boltzmann_temperature must be > 0, got {boltzmann_temperature}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.boltzmann_temperature = float(boltzmann_temperature)

    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        online_next_q_values = self.q_network(next_obs)
        target_next_q_values = self.target_network(next_obs)
        next_log_policy = _soft_action_log_probs(online_next_q_values, temperature=self.boltzmann_temperature)
        next_policy = next_log_policy.exp()
        return (next_policy * target_next_q_values).sum(dim=-1)


class AdvantageLearningDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        advantage_alpha: float,
    ) -> None:
        if float(advantage_alpha) < 0.0:
            raise ValueError(f"advantage_alpha must be >= 0, got {advantage_alpha}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.advantage_alpha = float(advantage_alpha)

    def _compute_target_q_values(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        del obs
        next_q_values = self._next_target_q_values(next_obs)
        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        state_values = q_values.max(dim=-1).values
        advantage_penalty = self.advantage_alpha * (state_values - chosen_q_values)
        return rewards + self.gamma * next_q_values * (1.0 - dones) - advantage_penalty


class PersistentAdvantageLearningDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        persistent_advantage_alpha: float,
    ) -> None:
        if float(persistent_advantage_alpha) < 0.0:
            raise ValueError(f"persistent_advantage_alpha must be >= 0, got {persistent_advantage_alpha}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.persistent_advantage_alpha = float(persistent_advantage_alpha)

    def _compute_target_q_values(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        del obs
        next_target_q_values = self.target_network(next_obs)
        next_state_values = next_target_q_values.max(dim=-1).values
        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        state_values = q_values.max(dim=-1).values
        current_action_gap = state_values - chosen_q_values
        next_action_values = next_target_q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        next_action_gap = (next_state_values - next_action_values) * (1.0 - dones)
        persistent_gap = torch.maximum(current_action_gap, next_action_gap)
        return (
            rewards + self.gamma * next_state_values * (1.0 - dones) - self.persistent_advantage_alpha * persistent_gap
        )


class MunchausenDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        munchausen_alpha: float,
        entropy_temperature: float,
        munchausen_clip_min: float,
    ) -> None:
        if float(munchausen_alpha) < 0.0:
            raise ValueError(f"munchausen_alpha must be >= 0, got {munchausen_alpha}")
        if float(entropy_temperature) <= 0.0:
            raise ValueError(f"entropy_temperature must be > 0, got {entropy_temperature}")
        if float(munchausen_clip_min) > 0.0:
            raise ValueError(f"munchausen_clip_min must be <= 0, got {munchausen_clip_min}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.munchausen_alpha = float(munchausen_alpha)
        self.entropy_temperature = float(entropy_temperature)
        self.munchausen_clip_min = float(munchausen_clip_min)

    def _next_soft_state_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_q_values = self.target_network(next_obs)
        return _soft_state_value(
            next_q_values,
            next_q_values,
            temperature=self.entropy_temperature,
        )

    def _compute_target_q_values(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        del obs
        current_log_policy = _soft_action_log_probs(q_values, temperature=self.entropy_temperature)
        munchausen_bonus = self.munchausen_alpha * current_log_policy.gather(1, actions.long().unsqueeze(-1)).squeeze(
            -1
        ).clamp(
            min=self.munchausen_clip_min,
            max=0.0,
        )
        next_soft_values = self._next_soft_state_values(next_obs)
        return rewards + munchausen_bonus + self.gamma * next_soft_values * (1.0 - dones)


class MunchausenDoubleDQN(MunchausenDQN):
    def _next_soft_state_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        online_next_q_values = self.q_network(next_obs)
        target_next_q_values = self.target_network(next_obs)
        return _soft_state_value(
            online_next_q_values,
            target_next_q_values,
            temperature=self.entropy_temperature,
        )


class CQLDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        cql_alpha: float,
    ) -> None:
        if float(cql_alpha) < 0.0:
            raise ValueError(f"cql_alpha must be >= 0, got {cql_alpha}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.cql_alpha = float(cql_alpha)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_network(obs)
        with torch.no_grad():
            target_q_values = self._compute_target_q_values(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                q_values=q_values,
            )

        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        td_errors = target_q_values - chosen_q_values
        self.last_td_errors = td_errors.detach().abs()

        td_loss = _weighted_smooth_l1_loss(chosen_q_values, target_q_values, weights=batch.get("weights"))
        cql_penalty = torch.logsumexp(q_values, dim=-1).mean() - chosen_q_values.mean()
        loss = td_loss + self.cql_alpha * cql_penalty

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "td_loss": float(td_loss.detach().cpu().item()),
            "cql_penalty": float(cql_penalty.detach().cpu().item()),
            "q_value_mean": float(chosen_q_values.mean().detach().cpu().item()),
            "target_mean": float(target_q_values.mean().detach().cpu().item()),
            "td_error_mean": float(td_errors.abs().mean().detach().cpu().item()),
        }
        return UpdateResult(metrics=metrics, num_gradient_steps=1)


class CQLDoubleDQN(CQLDQN):
    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        next_actions = self.q_network(next_obs).argmax(dim=-1, keepdim=True)
        return self.target_network(next_obs).gather(1, next_actions).squeeze(-1)


class ClippedDoubleDQN(DQN):
    def _next_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        online_next_q_values = self.q_network(next_obs)
        target_next_q_values = self.target_network(next_obs)
        next_actions = online_next_q_values.argmax(dim=-1, keepdim=True)
        online_selected_q_values = online_next_q_values.gather(1, next_actions).squeeze(-1)
        target_selected_q_values = target_next_q_values.gather(1, next_actions).squeeze(-1)
        return torch.minimum(online_selected_q_values, target_selected_q_values)


class HystereticDQN(DQN):
    def __init__(
        self,
        *,
        q_network: MLPQNetwork | MLPDuelingQNetwork | MLPNoisyQNetwork | MLPDuelingNoisyQNetwork,
        learning_rate: float,
        gamma: float,
        target_update_interval: int,
        hysteretic_beta: float,
    ) -> None:
        if not 0.0 <= float(hysteretic_beta) <= 1.0:
            raise ValueError(f"hysteretic_beta must be in [0, 1], got {hysteretic_beta}")
        super().__init__(
            q_network=q_network,
            learning_rate=learning_rate,
            gamma=gamma,
            target_update_interval=target_update_interval,
        )
        self.hysteretic_beta = float(hysteretic_beta)

    def update(self, batch: dict[str, Any], *, global_step: int) -> UpdateResult:
        self.set_train_mode()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_network(obs)
        with torch.no_grad():
            target_q_values = self._compute_target_q_values(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                q_values=q_values,
            )

        chosen_q_values = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        td_errors = target_q_values - chosen_q_values
        self.last_td_errors = td_errors.detach().abs()
        effective_td_errors = torch.where(td_errors >= 0.0, td_errors, self.hysteretic_beta * td_errors)
        effective_targets = (chosen_q_values + effective_td_errors).detach()
        loss = _weighted_smooth_l1_loss(chosen_q_values, effective_targets, weights=batch.get("weights"))

        terms = {
            "loss": loss,
            "q_value_mean": chosen_q_values.mean(),
            "target_mean": target_q_values.mean(),
            "td_error_mean": td_errors.abs().mean(),
            "effective_td_error_mean": effective_td_errors.abs().mean(),
        }

        self.optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        self.optimizer.step()

        if global_step % self.target_update_interval == 0:
            self.sync_target_network()

        metrics = {name: float(value.detach().cpu().item()) for name, value in terms.items()}
        return UpdateResult(metrics=metrics, num_gradient_steps=1)
