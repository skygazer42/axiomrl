import torch

from axiomrl.algorithms.awac import AWAC
from axiomrl.models import MLPSACModel


def test_awac_update_returns_named_metrics() -> None:
    algorithm = AWAC(
        model=MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(16, 16)),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        awac_lambda=1.0,
        max_advantage_weight=20.0,
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
