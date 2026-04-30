from pathlib import Path

import pytest
import torch

from axiomrl.data.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample() -> None:
    buffer = ReplayBuffer(capacity=5, obs_shape=(4,), action_shape=())

    for step in range(3):
        buffer.add(
            obs=torch.full((4,), float(step)),
            actions=step % 2,
            rewards=float(step),
            next_obs=torch.full((4,), float(step + 1)),
            dones=float(step == 2),
        )

    batch = buffer.sample(2)

    assert len(buffer) == 3
    assert batch["obs"].shape == (2, 4)
    assert batch["actions"].shape == (2,)
    assert batch["rewards"].shape == (2,)
    assert batch["next_obs"].shape == (2, 4)
    assert batch["dones"].shape == (2,)


def test_replay_buffer_state_roundtrip(tmp_path: Path) -> None:
    buffer = ReplayBuffer(capacity=4, obs_shape=(4,), action_shape=())
    buffer.add(
        obs=torch.ones(4),
        actions=1,
        rewards=1.0,
        next_obs=torch.ones(4) * 2,
        dones=0.0,
    )

    state = buffer.state_dict()
    restored = ReplayBuffer(capacity=4, obs_shape=(4,), action_shape=())
    restored.load_state_dict(state)

    assert len(restored) == 1
    sample = restored.sample(1)
    assert sample["actions"].item() == 1
    assert sample["rewards"].item() == pytest.approx(1.0)


def test_replay_buffer_state_dict_moves_tensor_payloads_to_cpu() -> None:
    buffer = ReplayBuffer(capacity=4, obs_shape=(4,), action_shape=(), device="cpu")
    buffer.add(
        obs=torch.ones(4),
        actions=1,
        rewards=1.0,
        next_obs=torch.ones(4) * 2,
        dones=0.0,
    )

    state = buffer.state_dict()

    assert state["obs"].device.type == "cpu"
    assert state["actions"].device.type == "cpu"
    assert state["rewards"].device.type == "cpu"
    assert state["next_obs"].device.type == "cpu"
    assert state["dones"].device.type == "cpu"
