import numpy as np
import pytest
import torch


def test_compute_discounted_returns_to_go_resets_at_episode_boundaries() -> None:
    from axiomrl.data.offline_dataset import compute_discounted_returns_to_go

    returns_to_go = compute_discounted_returns_to_go(
        rewards=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dones=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        gamma=0.5,
    )

    assert torch.allclose(returns_to_go, torch.tensor([2.0, 2.0, 5.0, 4.0]))


def test_transition_dataset_len_and_sample_from_numpy() -> None:
    from axiomrl.data.offline_dataset import TransitionDataset

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
    from axiomrl.data.offline_dataset import TransitionDataset

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
    from axiomrl.data.offline_dataset import TransitionDataset

    with pytest.raises(ValueError, match="same length"):
        TransitionDataset(
            obs=np.zeros((3, 4), dtype=np.float32),
            actions=np.zeros((4,), dtype=np.int64),
            rewards=np.zeros((3,), dtype=np.float32),
            next_obs=np.zeros((3, 4), dtype=np.float32),
            dones=np.zeros((3,), dtype=np.float32),
        )


def test_transition_dataset_preserves_continuous_action_shape() -> None:
    from axiomrl.data.offline_dataset import TransitionDataset

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


def test_transition_dataset_from_dict_accepts_terminations_without_dones() -> None:
    from axiomrl.data.offline_dataset import TransitionDataset

    dataset = TransitionDataset.from_dict(
        {
            "obs": np.zeros((3, 4), dtype=np.float32),
            "actions": np.zeros((3,), dtype=np.int64),
            "rewards": np.ones((3,), dtype=np.float32),
            "next_obs": np.ones((3, 4), dtype=np.float32),
            "terminations": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
    )

    assert len(dataset) == 3
    assert torch.equal(dataset.dones, torch.tensor([0.0, 1.0, 0.0]))


def test_transition_dataset_preserves_optional_next_actions() -> None:
    from axiomrl.data.offline_dataset import TransitionDataset

    dataset = TransitionDataset(
        obs=np.zeros((5, 3), dtype=np.float32),
        actions=np.zeros((5, 2), dtype=np.float32),
        rewards=np.ones((5,), dtype=np.float32),
        next_obs=np.ones((5, 3), dtype=np.float32),
        dones=np.zeros((5,), dtype=np.float32),
        next_actions=np.full((5, 2), 0.5, dtype=np.float32),
    )

    batch = dataset.sample(batch_size=3, device="cpu")
    transformed = dataset.with_reward_transform(scale=2.0)

    assert "next_actions" in batch
    assert batch["next_actions"].shape == (3, 2)
    assert transformed.next_actions is not None
    assert transformed.next_actions.shape == (5, 2)


def test_transition_dataset_can_compute_and_recompute_returns_to_go() -> None:
    from axiomrl.data.offline_dataset import TransitionDataset

    dataset = TransitionDataset(
        obs=np.zeros((4, 2), dtype=np.float32),
        actions=np.zeros((4, 1), dtype=np.float32),
        rewards=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        next_obs=np.ones((4, 2), dtype=np.float32),
        dones=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )

    with_returns = dataset.with_discounted_returns_to_go(gamma=0.5)
    transformed = dataset.with_reward_transform(scale=2.0, returns_to_go_gamma=0.5)
    batch = with_returns.sample(batch_size=2, device="cpu")

    assert with_returns.returns_to_go is not None
    assert torch.allclose(with_returns.returns_to_go, torch.tensor([2.0, 2.0, 5.0, 4.0]))
    assert "returns_to_go" in batch
    assert transformed.returns_to_go is not None
    assert torch.allclose(transformed.returns_to_go, torch.tensor([4.0, 4.0, 10.0, 8.0]))
