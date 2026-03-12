import pytest
import torch

from rl_training.algorithms.rebrac import ReBRAC, rebrac_loss
from rl_training.models.mlp_td3 import MLPTD3Model


def test_rebrac_loss_returns_named_metrics() -> None:
    metrics = rebrac_loss(
        {
            "q1_values": torch.zeros(8, dtype=torch.float32),
            "q2_values": torch.zeros(8, dtype=torch.float32),
            "target_q_values": torch.ones(8, dtype=torch.float32),
            "actor_q_values": torch.full((8,), 0.5, dtype=torch.float32),
            "actor_bc_loss": torch.ones((), dtype=torch.float32),
            "critic_bc_penalty": torch.full((), 0.25, dtype=torch.float32),
            "actor_bc_weight": torch.full((), 1.0, dtype=torch.float32),
            "critic_bc_weight": torch.full((), 1.0, dtype=torch.float32),
            "actor_q_weight": torch.full((), 1.0, dtype=torch.float32),
        }
    )

    assert set(metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "actor_bc_loss",
        "actor_q_mean",
        "critic_bc_penalty",
        "actor_bc_weight",
        "critic_bc_weight",
        "actor_q_weight",
    }


def test_rebrac_rejects_invalid_regularization_weights() -> None:
    model = MLPTD3Model(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="actor_bc_weight must be > 0"):
        ReBRAC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            actor_bc_weight=0.0,
            critic_bc_weight=1.0,
            actor_q_weight=1.0,
        )

    with pytest.raises(ValueError, match="critic_bc_weight must be >= 0"):
        ReBRAC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            actor_bc_weight=1.0,
            critic_bc_weight=-1.0,
            actor_q_weight=1.0,
        )


def test_rebrac_update_returns_named_metrics() -> None:
    torch.manual_seed(151)

    algorithm = ReBRAC(
        model=MLPTD3Model(obs_dim=3, action_dim=1, hidden_sizes=(32, 32)),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        actor_bc_weight=1.0,
        critic_bc_weight=1.0,
        actor_q_weight=1.0,
    )

    result = algorithm.update(
        {
            "obs": torch.randn((8, 3), dtype=torch.float32),
            "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
            "rewards": torch.randn(8, dtype=torch.float32),
            "next_obs": torch.randn((8, 3), dtype=torch.float32),
            "dones": torch.zeros(8, dtype=torch.float32),
            "next_actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "actor_bc_loss",
        "actor_q_mean",
        "critic_bc_penalty",
        "actor_bc_weight",
        "critic_bc_weight",
        "actor_q_weight",
    }
