import torch

from axiomrl.data.offline_dataset import TransitionDataset
from axiomrl.data.trajectory_windows import TrajectoryWindowDataset


def test_trajectory_window_dataset_builds_padded_episode_windows() -> None:
    dataset = TransitionDataset(
        obs=torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
        actions=torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32),
        rewards=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        next_obs=torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float32),
        dones=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32),
    ).with_discounted_returns_to_go(gamma=1.0)

    windows = TrajectoryWindowDataset.from_transition_dataset(dataset, context_length=3)

    assert len(windows) == 4
    assert windows.obs.shape == (4, 3, 1)
    assert windows.actions.shape == (4, 3, 1)
    assert windows.returns_to_go is not None
    assert windows.returns_to_go.shape == (4, 3)
    assert windows.timesteps.shape == (4, 3)
    assert windows.mask.shape == (4, 3)
    assert torch.equal(windows.mask[0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(windows.mask[1], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(windows.mask[2], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(windows.timesteps[1], torch.tensor([0, 0, 1], dtype=torch.int64))
    assert torch.equal(windows.timesteps[2], torch.tensor([0, 0, 0], dtype=torch.int64))


def test_trajectory_window_dataset_sample_returns_named_tensors() -> None:
    dataset = TransitionDataset(
        obs=torch.randn(6, 3),
        actions=torch.randn(6, 2),
        rewards=torch.randn(6),
        next_obs=torch.randn(6, 3),
        dones=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    ).with_discounted_returns_to_go(gamma=0.99)

    windows = TrajectoryWindowDataset.from_transition_dataset(dataset, context_length=4)
    batch = windows.sample(3)

    assert set(batch) == {"obs", "actions", "returns_to_go", "timesteps", "mask"}
    assert batch["obs"].shape == (3, 4, 3)
    assert batch["actions"].shape == (3, 4, 2)
    assert batch["returns_to_go"].shape == (3, 4)
    assert batch["timesteps"].shape == (3, 4)
    assert batch["mask"].shape == (3, 4)
