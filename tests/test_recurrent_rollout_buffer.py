import pytest
import torch

from rl_training.data import RecurrentRolloutBuffer


def test_recurrent_rollout_buffer_stores_hidden_state_and_computes_returns() -> None:
    buffer = RecurrentRolloutBuffer(
        num_steps=4,
        num_envs=2,
        obs_shape=(4,),
        hidden_size=8,
        num_layers=1,
    )

    for step in range(4):
        buffer.add(
            obs=torch.full((2, 4), fill_value=float(step)),
            actions=torch.tensor([0, 1]),
            rewards=torch.tensor([1.0, 0.5]),
            dones=torch.tensor([0.0, float(step == 3)]),
            episode_starts=torch.tensor([float(step == 0), float(step == 0)]),
            values=torch.tensor([0.2, 0.1]),
            logprobs=torch.tensor([-0.1, -0.2]),
            recurrent_state=(
                torch.full((1, 2, 8), fill_value=float(step)),
                torch.full((1, 2, 8), fill_value=float(step + 1)),
            ),
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2), gamma=0.99, gae_lambda=0.95)

    assert buffer.step == 4
    assert buffer.hidden_h.shape == (4, 1, 2, 8)
    assert buffer.hidden_c.shape == (4, 1, 2, 8)
    assert torch.isfinite(buffer.advantages).all()
    assert torch.isfinite(buffer.returns).all()


def test_recurrent_rollout_buffer_sequence_minibatches_include_masks_and_initial_state() -> None:
    buffer = RecurrentRolloutBuffer(
        num_steps=3,
        num_envs=1,
        obs_shape=(4,),
        hidden_size=8,
        num_layers=1,
    )

    for step in range(3):
        buffer.add(
            obs=torch.full((1, 4), fill_value=float(step)),
            actions=torch.tensor([step % 2]),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([float(step == 2)]),
            episode_starts=torch.tensor([float(step == 0)]),
            values=torch.tensor([0.1]),
            logprobs=torch.tensor([-0.1]),
            recurrent_state=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(1), gamma=0.99, gae_lambda=0.95)
    minibatches = list(
        buffer.iter_sequence_minibatches(
            sequence_length=2,
            sequences_per_batch=1,
            shuffle=False,
        )
    )

    assert len(minibatches) == 2
    assert minibatches[0]["obs"].shape == (2, 1, 4)
    assert minibatches[0]["initial_h"].shape == (1, 1, 8)
    assert minibatches[1]["mask"].shape == (2, 1)
    assert torch.equal(minibatches[1]["mask"][:, 0], torch.tensor([1.0, 0.0]))


def test_recurrent_rollout_buffer_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="num_steps must be > 0"):
        RecurrentRolloutBuffer(num_steps=0, num_envs=1, obs_shape=(4,), hidden_size=8)

    buffer = RecurrentRolloutBuffer(num_steps=2, num_envs=1, obs_shape=(4,), hidden_size=8)

    with pytest.raises(ValueError, match="sequence_length must be > 0"):
        next(buffer.iter_sequence_minibatches(sequence_length=0, sequences_per_batch=1, shuffle=False))

    with pytest.raises(ValueError, match="sequences_per_batch must be > 0"):
        next(buffer.iter_sequence_minibatches(sequence_length=1, sequences_per_batch=0, shuffle=False))
