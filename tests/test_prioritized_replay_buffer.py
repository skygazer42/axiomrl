import torch

from axiomrl.data.prioritized_replay_buffer import PrioritizedReplayBuffer


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


def test_prioritized_replay_buffer_state_dict_moves_tensor_payloads_to_cpu() -> None:
    buffer = PrioritizedReplayBuffer(
        capacity=8,
        obs_shape=(4,),
        action_shape=(),
        alpha=0.6,
        device="cpu",
    )
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
    assert state["priorities"].device.type == "cpu"
