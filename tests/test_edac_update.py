import pytest
import torch

from axiomrl.algorithms.edac import EDAC, critic_diversity_loss, edac_loss
from axiomrl.models.mlp_redq import MLPREDQModel


def test_critic_diversity_loss_returns_zero_for_orthogonal_gradients() -> None:
    gradients = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )

    loss = critic_diversity_loss(gradients)

    assert float(loss.item()) == pytest.approx(0.0)


def test_edac_loss_returns_named_metrics() -> None:
    metrics = edac_loss(
        {
            "critic_q_values": torch.zeros((8, 4), dtype=torch.float32),
            "target_q_values": torch.ones(8, dtype=torch.float32),
            "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
            "sampled_q_values": torch.zeros((8, 4), dtype=torch.float32),
            "alpha": 0.2,
            "diversity_loss": 0.5,
            "eta": 1.0,
        }
    )

    assert set(metrics) >= {
        "critic_loss",
        "critic_mse_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
        "diversity_loss",
        "q_data_std",
        "q_policy_std",
    }


def test_edac_rejects_invalid_alpha_and_eta() -> None:
    model = MLPREDQModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32), num_critics=4)

    with pytest.raises(ValueError, match="alpha must be > 0"):
        EDAC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            alpha=0.0,
            tau=0.005,
            num_critics=4,
            eta=1.0,
        )

    with pytest.raises(ValueError, match="eta must be >= 0"):
        EDAC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            alpha=0.2,
            tau=0.005,
            num_critics=4,
            eta=-1.0,
        )


def test_edac_update_returns_named_metrics() -> None:
    torch.manual_seed(43)

    algorithm = EDAC(
        model=MLPREDQModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32), num_critics=4),
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        num_critics=4,
        eta=1.0,
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
        "critic_mse_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
        "diversity_loss",
        "q_data_std",
        "q_policy_std",
    }
