import pytest
import torch

from rl_training.algorithms.iql import IQL, iql_loss
from rl_training.models.mlp_iql import MLPIQLModel


def test_mlp_iql_model_samples_bounded_actions_and_returns_q_and_v() -> None:
    model = MLPIQLModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
    )

    sampled = model.sample_actions(torch.zeros((4, 3), dtype=torch.float32))
    q1, q2 = model.q_values(torch.zeros((4, 3), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32))
    values = model.value(torch.zeros((4, 3), dtype=torch.float32))

    assert sampled.actions.shape == (4, 1)
    assert sampled.logprobs.shape == (4,)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert values.shape == (4,)
    assert torch.all(sampled.actions <= 1.0 + 1e-6)
    assert torch.all(sampled.actions >= -1.0 - 1e-6)


def test_iql_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "value_predictions": torch.zeros(8, dtype=torch.float32),
        "target_state_values": torch.ones(8, dtype=torch.float32),
        "behavior_logprobs": torch.zeros(8, dtype=torch.float32),
        "advantage_weights": torch.ones(8, dtype=torch.float32),
    }

    metrics = iql_loss(batch)

    assert set(metrics) >= {"critic_loss", "value_loss", "actor_loss", "target_q_mean", "advantage_weight_mean"}


def test_iql_rejects_invalid_expectile_and_beta() -> None:
    model = MLPIQLModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
    )

    with pytest.raises(ValueError, match="expectile must be in"):
        IQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            expectile=0.0,
            beta=3.0,
            max_advantage_weight=100.0,
        )

    with pytest.raises(ValueError, match="beta must be > 0"):
        IQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            expectile=0.7,
            beta=0.0,
            max_advantage_weight=100.0,
        )


def test_iql_update_returns_update_result() -> None:
    torch.manual_seed(31)

    model = MLPIQLModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
    )
    algorithm = IQL(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        expectile=0.7,
        beta=3.0,
        max_advantage_weight=100.0,
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
    assert set(result.metrics) >= {"critic_loss", "value_loss", "actor_loss", "target_q_mean", "advantage_weight_mean"}
