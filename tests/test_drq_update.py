import torch

from axiomrl.algorithms.drq import DrQ, drq_loss
from axiomrl.models import CNNDrQModel


def test_drq_loss_returns_named_metrics() -> None:
    metrics = drq_loss(
        {
            "q1_values": torch.zeros(4, dtype=torch.float32),
            "q2_values": torch.zeros(4, dtype=torch.float32),
            "target_q_values": torch.ones(4, dtype=torch.float32),
            "sampled_logprobs": torch.full((4,), -0.3, dtype=torch.float32),
            "sampled_q1": torch.full((4,), 0.4, dtype=torch.float32),
            "sampled_q2": torch.full((4,), 0.2, dtype=torch.float32),
            "alpha": torch.tensor(0.1, dtype=torch.float32),
        }
    )

    assert set(metrics) == {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}


def test_drq_update_returns_expected_metrics() -> None:
    model = CNNDrQModel(
        obs_shape=(9, 84, 84),
        action_dim=2,
        features_dim=64,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    algorithm = DrQ(
        model=model,
        learning_rate=1e-4,
        gamma=0.99,
        alpha=0.1,
        tau=0.01,
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
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term", "algorithm_updates"}


def test_drq_state_dict_round_trip_preserves_alpha_and_update_count() -> None:
    model = CNNDrQModel(
        obs_shape=(9, 84, 84),
        action_dim=2,
        features_dim=64,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    algorithm = DrQ(
        model=model,
        learning_rate=1e-4,
        gamma=0.99,
        alpha=0.1,
        tau=0.01,
        augmentation_pad=4,
    )
    batch = {
        "obs": torch.randint(0, 256, (8, 9, 84, 84), dtype=torch.uint8),
        "actions": torch.rand(8, 2, dtype=torch.float32) * 2.0 - 1.0,
        "rewards": torch.rand(8, dtype=torch.float32),
        "next_obs": torch.randint(0, 256, (8, 9, 84, 84), dtype=torch.uint8),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    algorithm.update(batch, global_step=8)
    state = algorithm.state_dict()

    restored = DrQ(
        model=CNNDrQModel(
            obs_shape=(9, 84, 84),
            action_dim=2,
            features_dim=64,
            actor_hidden_sizes=(32,),
            critic_hidden_sizes=(32,),
        ),
        learning_rate=1e-4,
        gamma=0.99,
        alpha=0.25,
        tau=0.01,
        augmentation_pad=4,
    )
    restored.load_state_dict(state)

    assert restored.alpha == algorithm.alpha
    assert restored.update_count == algorithm.update_count
