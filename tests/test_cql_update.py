import pytest
import torch

from rl_training.algorithms.cql import CQL, cql_loss
from rl_training.models.mlp_sac import MLPSACModel


def test_cql_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q1": torch.zeros(8, dtype=torch.float32),
        "sampled_q2": torch.zeros(8, dtype=torch.float32),
        "alpha": torch.as_tensor(0.2, dtype=torch.float32),
        "cql_penalty_q1": torch.ones((), dtype=torch.float32),
        "cql_penalty_q2": torch.ones((), dtype=torch.float32),
        "cql_alpha": torch.as_tensor(5.0, dtype=torch.float32),
    }

    metrics = cql_loss(batch)

    assert set(metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
        "cql_penalty",
    }


def test_cql_rejects_invalid_cql_alpha_and_num_cql_samples() -> None:
    model = MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="cql_alpha must be > 0"):
        CQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            alpha=0.2,
            tau=0.005,
            cql_alpha=0.0,
            num_cql_samples=10,
        )

    with pytest.raises(ValueError, match="num_cql_samples must be >= 1"):
        CQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            alpha=0.2,
            tau=0.005,
            cql_alpha=5.0,
            num_cql_samples=0,
        )


def test_cql_update_returns_update_result() -> None:
    torch.manual_seed(211)

    model = MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = CQL(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        cql_alpha=5.0,
        num_cql_samples=10,
    )

    batch = {
        "obs": torch.randn((8, 3), dtype=torch.float32),
        "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
        "rewards": torch.randn(8, dtype=torch.float32),
        "next_obs": torch.randn((8, 3), dtype=torch.float32),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
        "cql_penalty",
    }
