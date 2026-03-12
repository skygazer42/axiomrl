import pytest
import torch

from rl_training.models import LSTMActorCritic


def test_lstm_actor_critic_vector_obs_returns_actions_values_and_state() -> None:
    policy = LSTMActorCritic(
        obs_shape=(4,),
        action_dim=2,
        features_dim=32,
        encoder_hidden_sizes=(16,),
        head_hidden_sizes=(16,),
        hidden_size=32,
        num_layers=1,
    )

    rollout = policy.act(torch.zeros((2, 4), dtype=torch.float32))

    assert rollout.actions.shape == (2,)
    assert rollout.values.shape == (2,)
    assert rollout.logprobs.shape == (2,)
    assert rollout.state is not None
    assert rollout.state[0].shape == (1, 2, 32)
    assert rollout.state[1].shape == (1, 2, 32)


def test_lstm_actor_critic_image_obs_and_sequence_evaluation_work() -> None:
    policy = LSTMActorCritic(
        obs_shape=(4, 84, 84),
        action_dim=3,
        features_dim=64,
        head_hidden_sizes=(32,),
        hidden_size=32,
        num_layers=1,
    )

    single_rollout = policy.act(torch.zeros((4, 84, 84), dtype=torch.uint8))
    evaluated = policy.evaluate_actions_sequence(
        torch.zeros((3, 2, 4, 84, 84), dtype=torch.uint8),
        torch.zeros((3, 2), dtype=torch.int64),
        initial_state=policy.initial_state(2),
        episode_starts=torch.zeros((3, 2), dtype=torch.float32),
    )

    assert single_rollout.actions.shape == (1,)
    assert evaluated["logprobs"].shape == (3, 2)
    assert evaluated["entropy"].shape == (3, 2)
    assert evaluated["values"].shape == (3, 2)


def test_lstm_actor_critic_rejects_invalid_recurrent_settings() -> None:
    with pytest.raises(ValueError, match="hidden_size must be > 0"):
        LSTMActorCritic(obs_shape=(4,), action_dim=2, hidden_size=0)

    with pytest.raises(ValueError, match="num_layers must be > 0"):
        LSTMActorCritic(obs_shape=(4,), action_dim=2, num_layers=0)
