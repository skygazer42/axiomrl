import torch

from axiomrl.algorithms.drqv2 import DrQv2, drqv2_loss
from axiomrl.models import CNNDrQv2Model


def test_drqv2_loss_returns_named_metrics() -> None:
    metrics = drqv2_loss(
        {
            "q1_values": torch.zeros(4, dtype=torch.float32),
            "q2_values": torch.zeros(4, dtype=torch.float32),
            "target_q_values": torch.ones(4, dtype=torch.float32),
            "actor_q_values": torch.full((4,), 0.5, dtype=torch.float32),
        }
    )

    assert set(metrics) == {"critic_loss", "actor_loss", "target_q_mean"}


def test_drqv2_update_returns_expected_metrics() -> None:
    model = CNNDrQv2Model(
        obs_shape=(9, 84, 84),
        action_dim=2,
        features_dim=64,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    algorithm = DrQv2(
        model=model,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.01,
        policy_delay=2,
        augmentation_pad=4,
    )
    batch = {
        "obs": torch.randint(0, 256, (8, 9, 84, 84), dtype=torch.uint8),
        "actions": torch.rand(8, 2, dtype=torch.float32) * 2.0 - 1.0,
        "rewards": torch.rand(8, dtype=torch.float32),
        "next_obs": torch.randint(0, 256, (8, 9, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "algorithm_updates"}
