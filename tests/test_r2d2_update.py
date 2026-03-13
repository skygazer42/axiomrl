import torch

from rl_training.algorithms.r2d2 import R2D2, r2d2_loss
from rl_training.models import LSTMQNetwork


def _make_sequence_batch() -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros((4, 2, 4), dtype=torch.float32),
        "actions": torch.zeros((4, 2), dtype=torch.int64),
        "rewards": torch.ones((4, 2), dtype=torch.float32),
        "next_obs": torch.zeros((4, 2, 4), dtype=torch.float32),
        "dones": torch.zeros((4, 2), dtype=torch.float32),
        "episode_starts": torch.zeros((4, 2), dtype=torch.float32),
        "mask": torch.ones((4, 2), dtype=torch.float32),
        "initial_h": torch.zeros((1, 2, 32), dtype=torch.float32),
        "initial_c": torch.zeros((1, 2, 32), dtype=torch.float32),
        "weights": torch.ones((2,), dtype=torch.float32),
    }


def test_r2d2_loss_returns_named_metrics() -> None:
    batch = {
        "chosen_q_values": torch.zeros((4, 2), dtype=torch.float32),
        "target_q_values": torch.ones((4, 2), dtype=torch.float32),
        "mask": torch.ones((4, 2), dtype=torch.float32),
        "weights": torch.ones((2,), dtype=torch.float32),
    }

    metrics = r2d2_loss(batch)

    assert set(metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}


def test_r2d2_update_returns_update_result_and_sequence_priorities() -> None:
    algorithm = R2D2(
        q_network=LSTMQNetwork(
            obs_shape=(4,),
            action_dim=2,
            features_dim=32,
            encoder_hidden_sizes=(16,),
            head_hidden_sizes=(16,),
            hidden_size=32,
            num_layers=1,
        ),
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=2,
        priority_eta=0.9,
    )

    result = algorithm.update(_make_sequence_batch(), global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}
    assert algorithm.last_sequence_priorities is not None
    assert algorithm.last_sequence_priorities.shape == (2,)
