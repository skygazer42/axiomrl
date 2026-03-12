import pytest
import torch

from rl_training.algorithms.marwil import MARWIL, marwil_loss
from rl_training.models import MLPIQLModel


def test_marwil_loss_returns_named_metrics() -> None:
    metrics = marwil_loss(
        {
            "value_predictions": torch.zeros(4, dtype=torch.float32),
            "returns_to_go": torch.ones(4, dtype=torch.float32),
            "behavior_logprobs": torch.full((4,), -0.5, dtype=torch.float32),
            "advantages": torch.full((4,), 0.25, dtype=torch.float32),
            "advantage_weights": torch.full((4,), 1.5, dtype=torch.float32),
            "advantage_norm_scale": 10.0,
            "vf_coeff": 1.0,
        }
    )

    assert set(metrics) >= {
        "value_loss",
        "scaled_value_loss",
        "actor_loss",
        "total_loss",
        "returns_to_go_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
        "advantage_norm_scale",
    }


def test_marwil_rejects_invalid_hyperparameters() -> None:
    model = MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(16, 16))

    with pytest.raises(ValueError, match="beta must be >= 0"):
        MARWIL(
            model=model,
            learning_rate=3e-4,
            beta=-0.1,
            vf_coeff=1.0,
            moving_average_sqd_adv_norm_start=100.0,
            moving_average_sqd_adv_norm_update_rate=0.01,
        )

    with pytest.raises(ValueError, match="vf_coeff must be >= 0"):
        MARWIL(
            model=model,
            learning_rate=3e-4,
            beta=1.0,
            vf_coeff=-1.0,
            moving_average_sqd_adv_norm_start=100.0,
            moving_average_sqd_adv_norm_update_rate=0.01,
        )

    with pytest.raises(ValueError, match="moving_average_sqd_adv_norm_start must be > 0"):
        MARWIL(
            model=model,
            learning_rate=3e-4,
            beta=1.0,
            vf_coeff=1.0,
            moving_average_sqd_adv_norm_start=0.0,
            moving_average_sqd_adv_norm_update_rate=0.01,
        )

    with pytest.raises(ValueError, match="moving_average_sqd_adv_norm_update_rate must be in \\(0, 1\\]"):
        MARWIL(
            model=model,
            learning_rate=3e-4,
            beta=1.0,
            vf_coeff=1.0,
            moving_average_sqd_adv_norm_start=100.0,
            moving_average_sqd_adv_norm_update_rate=0.0,
        )


def test_marwil_update_returns_named_metrics_and_tracks_running_norm() -> None:
    algorithm = MARWIL(
        model=MLPIQLModel(obs_dim=3, action_dim=1, hidden_sizes=(16, 16)),
        learning_rate=3e-4,
        beta=1.0,
        vf_coeff=1.0,
        moving_average_sqd_adv_norm_start=100.0,
        moving_average_sqd_adv_norm_update_rate=0.5,
    )

    result = algorithm.update(
        {
            "obs": torch.zeros((8, 3), dtype=torch.float32),
            "actions": torch.zeros((8, 1), dtype=torch.float32),
            "returns_to_go": torch.ones((8,), dtype=torch.float32),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "value_loss",
        "scaled_value_loss",
        "actor_loss",
        "total_loss",
        "returns_to_go_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
        "advantage_norm_scale",
        "moving_average_sqd_adv_norm",
    }
    assert algorithm.state_dict()["moving_average_sqd_adv_norm"] > 0.0
