import numpy as np
import pytest
import torch


def test_transition_dataset_len_and_sample_from_numpy() -> None:
    from rl_training.data.offline_dataset import TransitionDataset

    obs = np.zeros((10, 4), dtype=np.float32)
    actions = (np.arange(10) % 2).astype(np.int64)
    rewards = np.ones((10,), dtype=np.float32)
    next_obs = np.ones((10, 4), dtype=np.float32)
    dones = np.zeros((10,), dtype=np.float32)

    dataset = TransitionDataset(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
    )

    batch = dataset.sample(batch_size=5, device="cpu")

    assert len(dataset) == 10
    assert set(batch) == {"obs", "actions", "rewards", "next_obs", "dones"}
    assert isinstance(batch["obs"], torch.Tensor)
    assert batch["obs"].shape == (5, 4)
    assert batch["actions"].shape == (5,)
    assert batch["rewards"].shape == (5,)
    assert batch["next_obs"].shape == (5, 4)
    assert batch["dones"].shape == (5,)
    assert batch["obs"].device.type == "cpu"


def test_transition_dataset_sample_raises_when_empty() -> None:
    from rl_training.data.offline_dataset import TransitionDataset

    dataset = TransitionDataset(
        obs=np.zeros((0, 4), dtype=np.float32),
        actions=np.zeros((0,), dtype=np.int64),
        rewards=np.zeros((0,), dtype=np.float32),
        next_obs=np.zeros((0, 4), dtype=np.float32),
        dones=np.zeros((0,), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="empty"):
        dataset.sample(batch_size=1)


def test_transition_dataset_validates_lengths() -> None:
    from rl_training.data.offline_dataset import TransitionDataset

    with pytest.raises(ValueError, match="same length"):
        TransitionDataset(
            obs=np.zeros((3, 4), dtype=np.float32),
            actions=np.zeros((4,), dtype=np.int64),
            rewards=np.zeros((3,), dtype=np.float32),
            next_obs=np.zeros((3, 4), dtype=np.float32),
            dones=np.zeros((3,), dtype=np.float32),
        )


def test_transition_dataset_preserves_continuous_action_shape() -> None:
    from rl_training.data.offline_dataset import TransitionDataset

    dataset = TransitionDataset(
        obs=np.zeros((10, 4), dtype=np.float32),
        actions=np.zeros((10, 2), dtype=np.float32),
        rewards=np.ones((10,), dtype=np.float32),
        next_obs=np.ones((10, 4), dtype=np.float32),
        dones=np.zeros((10,), dtype=np.float32),
    )

    batch = dataset.sample(batch_size=5, device="cpu")

    assert batch["actions"].shape == (5, 2)
    assert batch["actions"].dtype == torch.float32
