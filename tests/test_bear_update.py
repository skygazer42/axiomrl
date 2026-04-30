import pytest
import torch

from axiomrl.algorithms.bear import BEAR, bear_loss
from axiomrl.models.mlp_bear import MLPBEARModel


def test_bear_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "reconstruction_loss": torch.ones((), dtype=torch.float32),
        "kl_loss": torch.ones((), dtype=torch.float32),
        "behavior_kl_weight": torch.as_tensor(0.5, dtype=torch.float32),
        "actor_q_values": torch.zeros(8, dtype=torch.float32),
        "mmd_loss": torch.ones((), dtype=torch.float32),
        "mmd_alpha": torch.as_tensor(10.0, dtype=torch.float32),
    }

    metrics = bear_loss(batch)

    assert set(metrics) >= {
        "behavior_loss",
        "reconstruction_loss",
        "kl_loss",
        "critic_loss",
        "actor_loss",
        "mmd_loss",
        "target_q_mean",
    }


def test_bear_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        MLPBEARModel(
            obs_dim=3,
            action_dim=1,
            latent_dim=0,
            hidden_sizes=(32, 32),
        )

    model = MLPBEARModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_sizes=(32, 32),
    )

    with pytest.raises(ValueError, match="behavior_kl_weight must be >= 0"):
        BEAR(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            behavior_kl_weight=-1.0,
            mmd_sigma=20.0,
            mmd_alpha=10.0,
            num_mmd_action_samples=10,
        )

    with pytest.raises(ValueError, match="mmd_sigma must be > 0"):
        BEAR(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            behavior_kl_weight=0.5,
            mmd_sigma=0.0,
            mmd_alpha=10.0,
            num_mmd_action_samples=10,
        )

    with pytest.raises(ValueError, match="mmd_alpha must be > 0"):
        BEAR(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            behavior_kl_weight=0.5,
            mmd_sigma=20.0,
            mmd_alpha=0.0,
            num_mmd_action_samples=10,
        )

    with pytest.raises(ValueError, match="num_mmd_action_samples must be >= 1"):
        BEAR(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            behavior_kl_weight=0.5,
            mmd_sigma=20.0,
            mmd_alpha=10.0,
            num_mmd_action_samples=0,
        )


def test_bear_update_returns_update_result() -> None:
    torch.manual_seed(181)

    model = MLPBEARModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_sizes=(32, 32),
    )
    algorithm = BEAR(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        behavior_kl_weight=0.5,
        mmd_sigma=20.0,
        mmd_alpha=10.0,
        num_mmd_action_samples=10,
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
        "behavior_loss",
        "reconstruction_loss",
        "kl_loss",
        "critic_loss",
        "actor_loss",
        "mmd_loss",
        "target_q_mean",
    }
