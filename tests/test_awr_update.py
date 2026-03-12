import pytest
import torch

from rl_training.algorithms.awr import AWR, awr_loss
from rl_training.models.mlp_iql import MLPIQLModel


def test_awr_loss_returns_named_metrics() -> None:
    metrics = awr_loss(
        {
            "value_predictions": torch.zeros(8, dtype=torch.float32),
            "returns_to_go": torch.ones(8, dtype=torch.float32),
            "behavior_logprobs": torch.zeros(8, dtype=torch.float32),
            "advantages": torch.ones(8, dtype=torch.float32),
            "advantage_weights": torch.ones(8, dtype=torch.float32),
        }
    )

    assert set(metrics) >= {
        "value_loss",
        "actor_loss",
        "returns_to_go_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
    }


def test_awr_rejects_invalid_beta_and_max_weight() -> None:
    model = MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="beta must be > 0"):
        AWR(
            model=model,
            learning_rate=3e-4,
            beta=0.0,
            max_weight=20.0,
        )

    with pytest.raises(ValueError, match="max_weight must be > 0"):
        AWR(
            model=model,
            learning_rate=3e-4,
            beta=1.0,
            max_weight=0.0,
        )


def test_awr_update_returns_named_metrics() -> None:
    torch.manual_seed(111)

    algorithm = AWR(
        model=MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32)),
        learning_rate=3e-4,
        beta=1.0,
        max_weight=20.0,
    )

    result = algorithm.update(
        {
            "obs": torch.randn((8, 3), dtype=torch.float32),
            "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
            "returns_to_go": torch.abs(torch.randn(8, dtype=torch.float32)),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "value_loss",
        "actor_loss",
        "returns_to_go_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
    }

