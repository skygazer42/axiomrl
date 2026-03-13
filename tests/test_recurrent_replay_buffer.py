import torch

from rl_training.data import RecurrentReplayBuffer


def test_recurrent_replay_buffer_stores_chunks_with_masks_and_initial_state() -> None:
    buffer = RecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
    )

    for step in range(3):
        buffer.add(
            env_index=0,
            obs=torch.full((4,), float(step), dtype=torch.float32),
            actions=step % 2,
            rewards=float(step),
            next_obs=torch.full((4,), float(step + 1), dtype=torch.float32),
            dones=float(step == 2),
            episode_start=float(step == 0),
            initial_state=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
        )

    batch = buffer.sample(batch_size=1)

    assert batch["obs"].shape == (4, 1, 4)
    assert batch["actions"].shape == (4, 1)
    assert batch["mask"].shape == (4, 1)
    assert batch["initial_h"].shape == (1, 1, 8)
    assert batch["initial_c"].shape == (1, 1, 8)
    assert float(batch["mask"].sum().item()) == 3.0


def test_recurrent_replay_buffer_clear_active_chunks_reconciles_transition_count() -> None:
    buffer = RecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
    )

    for step in range(3):
        buffer.add(
            env_index=0,
            obs=torch.full((4,), float(step), dtype=torch.float32),
            actions=step % 2,
            rewards=float(step),
            next_obs=torch.full((4,), float(step + 1), dtype=torch.float32),
            dones=0.0,
            episode_start=float(step == 0),
            initial_state=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
        )

    assert buffer.num_transitions == 3

    buffer.clear_active_chunks()

    assert buffer.num_transitions == 0
