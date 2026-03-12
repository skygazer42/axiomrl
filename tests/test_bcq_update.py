import pytest
import torch

from rl_training.algorithms.bcq import BCQ, bcq_loss
from rl_training.models.mlp_bcq import MLPBCQModel


def test_bcq_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "reconstruction_loss": torch.ones((), dtype=torch.float32),
        "kl_loss": torch.ones((), dtype=torch.float32),
        "vae_kl_weight": torch.as_tensor(0.5, dtype=torch.float32),
        "actor_q_values": torch.zeros(8, dtype=torch.float32),
        "candidate_q_values": torch.zeros((8, 10), dtype=torch.float32),
    }

    metrics = bcq_loss(batch)

    assert set(metrics) >= {
        "vae_loss",
        "reconstruction_loss",
        "kl_loss",
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "candidate_q_mean",
    }


def test_bcq_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="latent_dim must be >= 1"):
        MLPBCQModel(
            obs_dim=3,
            action_dim=1,
            latent_dim=0,
            hidden_sizes=(32, 32),
        )

    with pytest.raises(ValueError, match="perturbation_scale must be > 0"):
        MLPBCQModel(
            obs_dim=3,
            action_dim=1,
            latent_dim=2,
            hidden_sizes=(32, 32),
            perturbation_scale=0.0,
        )

    model = MLPBCQModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_sizes=(32, 32),
    )

    with pytest.raises(ValueError, match="num_action_samples must be >= 1"):
        BCQ(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            num_action_samples=0,
            vae_kl_weight=0.5,
        )

    with pytest.raises(ValueError, match="vae_kl_weight must be >= 0"):
        BCQ(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            num_action_samples=10,
            vae_kl_weight=-1.0,
        )


def test_bcq_update_returns_update_result() -> None:
    torch.manual_seed(173)

    model = MLPBCQModel(
        obs_dim=3,
        action_dim=1,
        latent_dim=2,
        hidden_sizes=(32, 32),
        num_action_samples=10,
    )
    algorithm = BCQ(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        num_action_samples=10,
        vae_kl_weight=0.5,
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
        "vae_loss",
        "reconstruction_loss",
        "kl_loss",
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "candidate_q_mean",
    }
