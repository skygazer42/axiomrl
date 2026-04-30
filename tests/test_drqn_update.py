import torch

from axiomrl.algorithms.drqn import DRQN, drqn_loss
from axiomrl.models import LSTMQNetwork


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
    }


def test_lstm_q_network_vector_obs_returns_q_values_actions_and_state() -> None:
    network = LSTMQNetwork(
        obs_shape=(4,),
        action_dim=2,
        features_dim=32,
        encoder_hidden_sizes=(16,),
        head_hidden_sizes=(16,),
        hidden_size=32,
        num_layers=1,
    )

    rollout = network.act(torch.zeros((2, 4), dtype=torch.float32), epsilon=0.1)
    sequence_q_values = network.q_values_sequence(
        torch.zeros((4, 2, 4), dtype=torch.float32),
        initial_state=network.initial_state(2),
        episode_starts=torch.zeros((4, 2), dtype=torch.float32),
    )

    assert rollout.actions.shape == (2,)
    assert rollout.q_values.shape == (2, 2)
    assert rollout.state is not None
    assert rollout.state[0].shape == (1, 2, 32)
    assert sequence_q_values.shape == (4, 2, 2)


def test_drqn_loss_returns_named_metrics() -> None:
    batch = {
        "chosen_q_values": torch.zeros((4, 2), dtype=torch.float32),
        "target_q_values": torch.ones((4, 2), dtype=torch.float32),
        "mask": torch.ones((4, 2), dtype=torch.float32),
    }

    metrics = drqn_loss(batch)

    assert set(metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}


def test_drqn_update_returns_update_result() -> None:
    algorithm = DRQN(
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
    )

    result = algorithm.update(_make_sequence_batch(), global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}
