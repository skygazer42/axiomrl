import torch

from axiomrl.algorithms.crr import CRR, crr_loss
from axiomrl.models import MLPSACModel


def test_crr_loss_returns_named_metrics() -> None:
    metrics = crr_loss(
        {
            "q1_values": torch.zeros(4, dtype=torch.float32),
            "q2_values": torch.zeros(4, dtype=torch.float32),
            "target_q_values": torch.ones(4, dtype=torch.float32),
            "behavior_logprobs": torch.full((4,), -0.5, dtype=torch.float32),
            "advantages": torch.full((4,), 0.25, dtype=torch.float32),
            "advantage_weights": torch.full((4,), 1.5, dtype=torch.float32),
        }
    )

    assert set(metrics) == {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
    }


def test_crr_update_returns_named_metrics() -> None:
    algorithm = CRR(
        model=MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(16, 16)),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        beta=1.0,
        n_action_samples=4,
        max_weight=20.0,
        advantage_type="mean",
        weight_type="exp",
    )

    result = algorithm.update(
        {
            "obs": torch.zeros((8, 3), dtype=torch.float32),
            "actions": torch.zeros((8, 1), dtype=torch.float32),
            "rewards": torch.zeros((8,), dtype=torch.float32),
            "next_obs": torch.zeros((8, 3), dtype=torch.float32),
            "dones": torch.zeros((8,), dtype=torch.float32),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "advantage_mean",
        "advantage_weight_mean",
        "behavior_logprob_mean",
    }
