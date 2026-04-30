import torch

from axiomrl.algorithms.c51_dqn import C51DQN, c51_loss
from axiomrl.models.mlp_c51_q_network import MLPC51QNetwork


def test_mlp_c51_q_network_forward_shape() -> None:
    network = MLPC51QNetwork(
        obs_dim=4,
        action_dim=2,
        v_min=0.0,
        v_max=200.0,
        num_atoms=51,
        hidden_sizes=(32, 32),
    )

    logits = network(torch.zeros((5, 4), dtype=torch.float32))

    assert logits.shape == (5, 2, 51)


def test_c51_loss_returns_named_metrics() -> None:
    batch = {
        "logits": torch.zeros((8, 2, 51), dtype=torch.float32),
        "actions": torch.zeros(8, dtype=torch.int64),
        "target_distributions": torch.ones((8, 51), dtype=torch.float32) / 51.0,
    }

    metrics = c51_loss(batch)

    assert set(metrics) >= {"loss"}


def test_c51_dqn_update_returns_update_result() -> None:
    torch.manual_seed(7)

    network = MLPC51QNetwork(
        obs_dim=4,
        action_dim=2,
        v_min=0.0,
        v_max=200.0,
        num_atoms=51,
        hidden_sizes=(32, 32),
    )
    algorithm = C51DQN(
        q_network=network,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_interval=2,
        v_min=0.0,
        v_max=200.0,
        num_atoms=51,
    )

    batch = {
        "obs": torch.randn((8, 4), dtype=torch.float32),
        "actions": torch.randint(0, 2, (8,), dtype=torch.int64),
        "rewards": torch.randn(8, dtype=torch.float32),
        "next_obs": torch.randn((8, 4), dtype=torch.float32),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert "loss" in result.metrics
