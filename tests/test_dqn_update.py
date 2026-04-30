import pytest
import torch
from torch import nn

from axiomrl.algorithms.dqn import DQN, dqn_loss
from axiomrl.models.mlp_dueling_noisy_q_network import MLPDuelingNoisyQNetwork
from axiomrl.models.mlp_dueling_q_network import MLPDuelingQNetwork
from axiomrl.models.mlp_noisy_q_network import MLPNoisyQNetwork
from axiomrl.models.mlp_q_network import MLPQNetwork


class FixedQNetwork(nn.Module):
    def __init__(self, weights: list[list[float]]) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor @ self.weights


def test_mlp_q_network_forward_shape() -> None:
    network = MLPQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    q_values = network(torch.zeros((5, 4), dtype=torch.float32))

    assert q_values.shape == (5, 2)


def test_mlp_dueling_q_network_forward_shape() -> None:
    network = MLPDuelingQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    q_values = network(torch.zeros((5, 4), dtype=torch.float32))

    assert q_values.shape == (5, 2)


def test_mlp_noisy_q_network_forward_shape() -> None:
    network = MLPNoisyQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    q_values = network(torch.zeros((5, 4), dtype=torch.float32))

    assert q_values.shape == (5, 2)


def test_mlp_dueling_noisy_q_network_forward_shape() -> None:
    network = MLPDuelingNoisyQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    q_values = network(torch.zeros((5, 4), dtype=torch.float32))

    assert q_values.shape == (5, 2)


def test_dqn_loss_returns_named_metrics() -> None:
    batch = {
        "q_values": torch.zeros((8, 2), dtype=torch.float32),
        "actions": torch.zeros(8, dtype=torch.int64),
        "target_q_values": torch.ones(8, dtype=torch.float32),
    }

    metrics = dqn_loss(batch)

    assert set(metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}


def test_dqn_update_returns_update_result() -> None:
    torch.manual_seed(3)

    network = MLPQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = DQN(
        q_network=network,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_interval=2,
    )

    batch = {
        "obs": torch.randn((8, 4), dtype=torch.float32),
        "actions": torch.randint(0, 2, (8,), dtype=torch.int64),
        "rewards": torch.ones(8, dtype=torch.float32),
        "next_obs": torch.randn((8, 4), dtype=torch.float32),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}


def test_double_dqn_uses_online_argmax_for_bootstrap_target() -> None:
    online_network = FixedQNetwork(
        [
            [5.0, 0.0],
            [0.0, 1.0],
        ]
    )
    algorithm = DQN(
        q_network=online_network,
        learning_rate=0.0,
        gamma=1.0,
        target_update_interval=100,
        double_q=True,
    )
    algorithm.target_network.weights.data = torch.tensor(
        [
            [1.0, 7.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    batch = {
        "obs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "actions": torch.tensor([0], dtype=torch.int64),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
        "next_obs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "dones": torch.tensor([0.0], dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=1)

    assert result.metrics["target_mean"] == pytest.approx(2.0)
