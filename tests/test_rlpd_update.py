import torch

from axiomrl.algorithms.rlpd import RLPD, rlpd_loss
from axiomrl.models.mlp_sac import MLPSACModel


def test_rlpd_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q1": torch.zeros(8, dtype=torch.float32),
        "sampled_q2": torch.zeros(8, dtype=torch.float32),
        "alpha": 0.2,
    }

    metrics = rlpd_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}


def test_rlpd_update_returns_named_metrics() -> None:
    torch.manual_seed(71)

    algorithm = RLPD(
        model=MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32)),
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
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
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}
