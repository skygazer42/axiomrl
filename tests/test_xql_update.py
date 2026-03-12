import pytest
import torch

from rl_training.algorithms.xql import XQL, gumbel_rescale_loss, xql_loss, xql_value_loss
from rl_training.models.mlp_iql import MLPIQLModel


def test_gumbel_rescale_loss_returns_finite_values() -> None:
    loss = gumbel_rescale_loss(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32),
        alpha=1.0,
        clip_max=5.0,
    )

    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()


def test_xql_loss_returns_named_metrics() -> None:
    metrics = xql_loss(
        {
            "q1_values": torch.zeros(8, dtype=torch.float32),
            "q2_values": torch.zeros(8, dtype=torch.float32),
            "target_q_values": torch.ones(8, dtype=torch.float32),
            "value_predictions": torch.zeros(8, dtype=torch.float32),
            "target_state_values": torch.ones(8, dtype=torch.float32),
            "behavior_logprobs": torch.zeros(8, dtype=torch.float32),
            "advantage_weights": torch.ones(8, dtype=torch.float32),
            "loss_temperature": 1.0,
            "max_value_diff_exp": 5.0,
        }
    )
    value_metrics = xql_value_loss(
        {
            "value_predictions": torch.zeros(8, dtype=torch.float32),
            "target_state_values": torch.ones(8, dtype=torch.float32),
            "loss_temperature": 1.0,
            "max_value_diff_exp": 5.0,
        }
    )

    assert set(metrics) >= {
        "critic_loss",
        "value_loss",
        "actor_loss",
        "target_q_mean",
        "advantage_weight_mean",
        "value_advantage_mean",
    }
    assert set(value_metrics) >= {
        "value_loss",
        "value_advantage_mean",
        "value_prediction_mean",
        "target_state_value_mean",
    }


def test_xql_rejects_invalid_loss_controls() -> None:
    model = MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="beta must be > 0"):
        XQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            beta=0.0,
            loss_temperature=1.0,
            max_advantage_weight=100.0,
        )

    with pytest.raises(ValueError, match="loss_temperature must be > 0"):
        XQL(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            beta=3.0,
            loss_temperature=0.0,
            max_advantage_weight=100.0,
        )


def test_xql_update_returns_named_metrics() -> None:
    torch.manual_seed(317)

    algorithm = XQL(
        model=MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32)),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        beta=3.0,
        loss_temperature=1.0,
        max_advantage_weight=100.0,
        max_value_diff_exp=5.0,
    )

    result = algorithm.update(
        {
            "obs": torch.randn((8, 3), dtype=torch.float32),
            "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
            "rewards": torch.randn(8, dtype=torch.float32),
            "next_obs": torch.randn((8, 3), dtype=torch.float32),
            "dones": torch.zeros(8, dtype=torch.float32),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "value_loss",
        "actor_loss",
        "target_q_mean",
        "advantage_weight_mean",
        "value_advantage_mean",
        "value_prediction_mean",
        "target_state_value_mean",
    }
