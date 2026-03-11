import pytest
import torch

from rl_training.algorithms.td3_bc import TD3BC, td3_bc_loss
from rl_training.models.mlp_td3 import MLPTD3Model


def test_td3_bc_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "actor_q_values": torch.zeros(8, dtype=torch.float32),
        "bc_loss": torch.ones((), dtype=torch.float32),
        "bc_lambda": torch.ones((), dtype=torch.float32),
    }

    metrics = td3_bc_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "bc_loss", "bc_lambda"}


def test_td3_bc_rejects_invalid_bc_alpha_and_policy_delay() -> None:
    model = MLPTD3Model(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="bc_alpha must be > 0"):
        TD3BC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            bc_alpha=0.0,
        )

    with pytest.raises(ValueError, match="policy_delay must be >= 1"):
        TD3BC(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=0,
            bc_alpha=2.5,
        )


def test_td3_bc_update_returns_update_result() -> None:
    torch.manual_seed(149)

    model = MLPTD3Model(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = TD3BC(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        bc_alpha=2.5,
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
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "bc_loss", "bc_lambda"}
