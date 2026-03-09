import torch

from rl_training.algorithms.dqn import DQN, dqn_loss
from rl_training.models.mlp_q_network import MLPQNetwork


def test_mlp_q_network_forward_shape() -> None:
    network = MLPQNetwork(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

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
