import torch

from axiomrl.data import PrioritizedRecurrentReplayBuffer


def test_prioritized_recurrent_replay_buffer_can_add_sample_and_update_priorities() -> None:
    buffer = PrioritizedRecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
        alpha=0.6,
        device="cpu",
    )

    for step in range(4):
        buffer.add(
            env_index=0,
            obs=torch.full((4,), float(step), dtype=torch.float32),
            actions=step % 2,
            rewards=float(step),
            next_obs=torch.full((4,), float(step + 1), dtype=torch.float32),
            dones=float(step == 3),
            episode_start=float(step == 0),
            initial_state=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
        )

    batch = buffer.sample(batch_size=1, beta=0.4)

    assert set(batch) >= {"obs", "actions", "mask", "initial_h", "initial_c", "indices", "weights"}
    assert batch["obs"].shape == (4, 1, 4)
    assert batch["actions"].shape == (4, 1)
    assert batch["weights"].shape == (1,)

    buffer.update_priorities(batch["indices"], torch.tensor([2.0], dtype=torch.float32))


def test_prioritized_recurrent_replay_buffer_sample_defaults_beta_to_zero() -> None:
    buffer = PrioritizedRecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
        alpha=0.6,
        device="cpu",
    )

    for step in range(4):
        buffer.add(
            env_index=0,
            obs=torch.full((4,), float(step), dtype=torch.float32),
            actions=step % 2,
            rewards=float(step),
            next_obs=torch.full((4,), float(step + 1), dtype=torch.float32),
            dones=float(step == 3),
            episode_start=float(step == 0),
            initial_state=(torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8))),
        )

    batch = buffer.sample(batch_size=1)

    assert batch["weights"].shape == (1,)


def test_prioritized_recurrent_replay_buffer_restores_sampling_hyperparameters_from_state() -> None:
    source = PrioritizedRecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
        alpha=0.7,
        priority_eps=1e-4,
        device="cpu",
    )
    target = PrioritizedRecurrentReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(4,),
        sequence_length=4,
        hidden_size=8,
        num_layers=1,
        alpha=0.1,
        priority_eps=1e-6,
        device="cpu",
    )

    target.load_state_dict(source.state_dict())

    assert target.alpha == source.alpha
    assert target.priority_eps == source.priority_eps
