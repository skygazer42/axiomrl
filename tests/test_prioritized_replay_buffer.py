import torch

from rl_training.data.prioritized_replay_buffer import PrioritizedReplayBuffer


def test_prioritized_replay_buffer_can_add_sample_and_update_priorities() -> None:
    buffer = PrioritizedReplayBuffer(
        capacity=32,
        obs_shape=(4,),
        action_shape=(),
        alpha=0.6,
        device="cpu",
    )

    for i in range(16):
        buffer.add(
            obs=[float(i), 0.0, 0.0, 0.0],
            actions=int(i % 2),
            rewards=1.0,
            next_obs=[float(i + 1), 0.0, 0.0, 0.0],
            dones=0.0,
        )

    batch = buffer.sample(batch_size=8, beta=0.4)

    assert set(batch) >= {"obs", "actions", "rewards", "next_obs", "dones", "indices", "weights"}
    assert batch["obs"].shape == (8, 4)
    assert batch["actions"].shape == (8,)
    assert batch["weights"].shape == (8,)

    indices = batch["indices"]
    priorities = torch.ones_like(indices, dtype=torch.float32) * 2.0
    buffer.update_priorities(indices, priorities)
