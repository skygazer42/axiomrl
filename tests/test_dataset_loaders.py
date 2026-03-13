from pathlib import Path

import numpy as np
import pytest
import torch

from rl_training.data import (
    TransitionDataset,
    compute_discounted_returns_to_go,
    load_transition_dataset,
    mix_transition_datasets,
)
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.iql_trainer import _build_offline_dataset


def test_transition_dataset_can_be_created_from_mapping() -> None:
    payload = {
        "obs": np.zeros((4, 3), dtype=np.float32),
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.ones((4,), dtype=np.float32),
        "next_obs": np.ones((4, 3), dtype=np.float32),
        "dones": np.zeros((4,), dtype=np.float32),
        "next_actions": np.full((4, 2), 0.25, dtype=np.float32),
        "returns_to_go": np.full((4,), 3.0, dtype=np.float32),
    }

    dataset = TransitionDataset.from_dict(payload)

    assert len(dataset) == 4
    assert dataset.actions.shape == (4, 2)
    assert dataset.next_actions is not None
    assert dataset.next_actions.shape == (4, 2)
    assert dataset.returns_to_go is not None
    assert dataset.returns_to_go.shape == (4,)


def test_load_transition_dataset_reads_npz_and_combines_episode_end_flags(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    np.savez(
        dataset_path,
        obs=np.zeros((5, 3), dtype=np.float32),
        actions=np.zeros((5, 1), dtype=np.float32),
        rewards=np.arange(5, dtype=np.float32),
        next_obs=np.ones((5, 3), dtype=np.float32),
        terminations=np.array([0, 0, 1, 0, 0], dtype=np.float32),
        truncations=np.array([0, 0, 0, 0, 1], dtype=np.float32),
    )

    dataset = load_transition_dataset("npz", dataset_path=dataset_path)

    assert len(dataset) == 5
    assert torch.equal(dataset.dones, torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))


def test_load_transition_dataset_rejects_torch_payloads(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pt"
    torch.save(
        {
            "obs": torch.zeros((6, 4), dtype=torch.float32),
            "actions": torch.zeros((6, 2), dtype=torch.float32),
            "rewards": torch.ones((6,), dtype=torch.float32),
            "next_obs": torch.ones((6, 4), dtype=torch.float32),
            "dones": torch.zeros((6,), dtype=torch.float32),
            "next_actions": torch.full((6, 2), 0.5, dtype=torch.float32),
            "returns_to_go": torch.full((6,), 2.5, dtype=torch.float32),
        },
        dataset_path,
    )

    with pytest.raises(ValueError, match="unsafe for untrusted datasets"):
        load_transition_dataset("pt", dataset_path=dataset_path)


def test_load_transition_dataset_rejects_torch_aliases(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pt"
    torch.save(
        {
            "obs": torch.zeros((2, 3), dtype=torch.float32),
            "actions": torch.zeros((2, 1), dtype=torch.float32),
            "rewards": torch.ones((2,), dtype=torch.float32),
            "next_obs": torch.ones((2, 3), dtype=torch.float32),
            "dones": torch.zeros((2,), dtype=torch.float32),
        },
        dataset_path,
    )

    for dataset_kind in ("pt", "pth", "torch"):
        with pytest.raises(ValueError, match="unsafe for untrusted datasets"):
            load_transition_dataset(dataset_kind, dataset_path=dataset_path)


def test_transition_dataset_reward_transform_scales_shifts_and_clips() -> None:
    dataset = TransitionDataset(
        obs=np.zeros((4, 2), dtype=np.float32),
        actions=np.zeros((4, 1), dtype=np.float32),
        rewards=np.array([-2.0, -1.0, 1.0, 4.0], dtype=np.float32),
        next_obs=np.ones((4, 2), dtype=np.float32),
        dones=np.zeros((4,), dtype=np.float32),
    )

    transformed = dataset.with_reward_transform(scale=0.5, shift=1.0, clip_min=0.0, clip_max=2.0)

    assert torch.equal(transformed.rewards, torch.tensor([0.0, 0.5, 1.5, 2.0]))


def test_mix_transition_datasets_respects_total_size_and_seed() -> None:
    returns_a = compute_discounted_returns_to_go(
        np.ones((6,), dtype=np.float32),
        np.zeros((6,), dtype=np.float32),
        gamma=0.9,
    )
    returns_b = compute_discounted_returns_to_go(
        np.full((6,), 10.0, dtype=np.float32),
        np.ones((6,), dtype=np.float32),
        gamma=0.9,
    )
    dataset_a = TransitionDataset(
        obs=np.zeros((6, 2), dtype=np.float32),
        actions=np.zeros((6, 1), dtype=np.float32),
        rewards=np.ones((6,), dtype=np.float32),
        next_obs=np.zeros((6, 2), dtype=np.float32),
        dones=np.zeros((6,), dtype=np.float32),
        next_actions=np.full((6, 1), 0.25, dtype=np.float32),
        returns_to_go=returns_a,
    )
    dataset_b = TransitionDataset(
        obs=np.full((6, 2), 5.0, dtype=np.float32),
        actions=np.ones((6, 1), dtype=np.float32),
        rewards=np.full((6,), 10.0, dtype=np.float32),
        next_obs=np.full((6, 2), 5.0, dtype=np.float32),
        dones=np.ones((6,), dtype=np.float32),
        returns_to_go=returns_b,
    )

    mixed_a = mix_transition_datasets(
        (dataset_a, dataset_b),
        weights=(0.75, 0.25),
        total_size=12,
        seed=7,
    )
    mixed_b = mix_transition_datasets(
        (dataset_a, dataset_b),
        weights=(0.75, 0.25),
        total_size=12,
        seed=7,
    )

    assert len(mixed_a) == 12
    assert torch.equal(mixed_a.obs, mixed_b.obs)
    assert torch.equal(mixed_a.rewards, mixed_b.rewards)
    assert set(torch.unique(mixed_a.rewards, dim=0).tolist()) <= {1.0, 10.0}
    assert mixed_a.next_actions is not None
    assert torch.equal(mixed_a.next_actions, mixed_b.next_actions)
    assert mixed_a.returns_to_go is not None
    assert torch.equal(mixed_a.returns_to_go, mixed_b.returns_to_go)


def test_build_offline_dataset_supports_npz_files_and_processing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pendulum_dataset.npz"
    np.savez(
        dataset_path,
        obs=np.zeros((8, 3), dtype=np.float32),
        actions=np.full((8, 1), 1.5, dtype=np.float32),
        next_actions=np.full((8, 1), -1.5, dtype=np.float32),
        rewards=np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        next_obs=np.ones((8, 3), dtype=np.float32),
        dones=np.zeros((8,), dtype=np.float32),
    )

    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=11,
        total_timesteps=8,
        output_dir=tmp_path / "runs",
        algo_kwargs={
            "dataset_kind": "npz",
            "dataset_path": str(dataset_path),
            "reward_scale": 0.5,
            "reward_shift": 1.0,
            "reward_clip_min": -0.5,
            "reward_clip_max": 2.0,
        },
    )

    dataset = _build_offline_dataset(config)

    assert len(dataset) == 8
    assert dataset.actions.dtype == torch.float32
    assert float(dataset.actions.abs().max().item()) <= 1.0
    assert dataset.next_actions is not None
    assert float(dataset.next_actions.abs().max().item()) <= 1.0
    assert float(dataset.rewards.min().item()) >= -0.5
    assert float(dataset.rewards.max().item()) <= 2.0


def test_build_random_offline_dataset_populates_next_actions(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=29,
        total_timesteps=8,
        output_dir=tmp_path / "runs",
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 32,
            "dataset_seed": 41,
        },
    )

    dataset = _build_offline_dataset(config)

    assert dataset.next_actions is not None
    assert dataset.next_actions.shape == dataset.actions.shape
    assert float(dataset.next_actions.abs().max().item()) <= 1.0


def test_build_offline_dataset_supports_dataset_mix(tmp_path: Path) -> None:
    dataset_path_a = tmp_path / "mix_a.npz"
    dataset_path_b = tmp_path / "mix_b.npz"
    np.savez(
        dataset_path_a,
        obs=np.zeros((4, 3), dtype=np.float32),
        actions=np.full((4, 1), 1.5, dtype=np.float32),
        rewards=np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
        next_obs=np.ones((4, 3), dtype=np.float32),
        dones=np.zeros((4,), dtype=np.float32),
    )
    np.savez(
        dataset_path_b,
        obs=np.full((4, 3), 3.0, dtype=np.float32),
        actions=np.full((4, 1), -1.5, dtype=np.float32),
        rewards=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        next_obs=np.full((4, 3), 2.0, dtype=np.float32),
        dones=np.ones((4,), dtype=np.float32),
    )

    config = TrainConfig(
        algo="iql",
        env_id="Pendulum-v1",
        seed=23,
        total_timesteps=8,
        output_dir=tmp_path / "runs",
        algo_kwargs={
            "dataset_mix": [
                {"kind": "npz", "dataset_path": str(dataset_path_a), "weight": 0.25},
                {"kind": "npz", "dataset_path": str(dataset_path_b), "weight": 0.75},
            ],
            "dataset_mix_size": 10,
            "dataset_mix_seed": 31,
            "reward_scale": 0.5,
            "reward_clip_min": -1.0,
            "reward_clip_max": 2.0,
        },
    )

    dataset = _build_offline_dataset(config)

    assert len(dataset) == 10
    assert float(dataset.actions.abs().max().item()) <= 1.0
    assert float(dataset.rewards.min().item()) >= -1.0
    assert float(dataset.rewards.max().item()) <= 2.0
