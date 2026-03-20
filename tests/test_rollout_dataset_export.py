from pathlib import Path

from rl_training.data import export_random_transition_dataset, load_transition_dataset


def test_export_random_transition_dataset_writes_valid_npz(tmp_path: Path) -> None:
    dataset_path = export_random_transition_dataset(
        "Pendulum-v1",
        tmp_path / "pendulum_rollout.npz",
        num_steps=32,
        seed=17,
    )

    dataset = load_transition_dataset("npz", dataset_path=dataset_path)

    assert dataset_path.exists()
    assert len(dataset) == 32
    assert tuple(dataset.obs.shape) == (32, 3)
    assert tuple(dataset.actions.shape) == (32, 1)
    assert dataset.next_actions is not None
    assert tuple(dataset.next_actions.shape) == (32, 1)
