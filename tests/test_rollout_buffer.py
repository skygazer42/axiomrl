import pytest
import torch

from axiomrl.data.rollout_buffer import RolloutBuffer


def test_rollout_buffer_computes_returns_and_advantages() -> None:
    buffer = RolloutBuffer(num_steps=2, num_envs=1, obs_shape=(4,), action_shape=())
    buffer.rewards[:] = torch.tensor([[1.0], [1.0]])
    buffer.values[:] = torch.tensor([[0.5], [0.5]])
    buffer.dones[:] = torch.tensor([[0.0], [0.0]])

    buffer.compute_returns_and_advantages(
        last_values=torch.tensor([0.0]),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert buffer.advantages.shape == (2, 1)
    assert buffer.returns.shape == (2, 1)
    assert buffer.advantages[0, 0].item() == pytest.approx(1.46525, rel=1e-5)
    assert buffer.advantages[1, 0].item() == pytest.approx(0.5, rel=1e-5)
    assert buffer.returns[0, 0].item() == pytest.approx(1.96525, rel=1e-5)
    assert buffer.returns[1, 0].item() == pytest.approx(1.0, rel=1e-5)


def test_rollout_buffer_iter_minibatches_flattens_storage() -> None:
    buffer = RolloutBuffer(num_steps=2, num_envs=2, obs_shape=(4,), action_shape=())
    buffer.obs[:] = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    buffer.actions[:] = torch.tensor([[0, 1], [1, 0]])
    buffer.logprobs[:] = torch.zeros((2, 2), dtype=torch.float32)
    buffer.values[:] = torch.ones((2, 2), dtype=torch.float32)
    buffer.advantages[:] = torch.ones((2, 2), dtype=torch.float32)
    buffer.returns[:] = torch.full((2, 2), 2.0, dtype=torch.float32)

    minibatches = list(buffer.iter_minibatches(minibatch_size=2, shuffle=False))

    assert len(minibatches) == 2
    assert minibatches[0]["obs"].shape == (2, 4)
    assert minibatches[0]["actions"].shape == (2,)
    assert minibatches[0]["returns"].shape == (2,)
