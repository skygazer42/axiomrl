import torch

from rl_training.algorithms.crossq import CrossQ, crossq_loss
from rl_training.models.mlp_crossq import MLPCrossQModel


def test_crossq_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q1": torch.zeros(8, dtype=torch.float32),
        "sampled_q2": torch.zeros(8, dtype=torch.float32),
        "alpha": torch.as_tensor(0.1, dtype=torch.float32),
    }

    metrics = crossq_loss(batch)

    assert set(metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
    }


def test_crossq_update_returns_update_result() -> None:
    torch.manual_seed(37)

    model = MLPCrossQModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        critic_hidden_sizes=(32, 32),
        bn_momentum=0.99,
    )
    algorithm = CrossQ(
        model=model,
        learning_rate=1e-3,
        gamma=0.99,
        alpha=0.1,
        policy_delay=1,
        adam_beta1=0.5,
    )

    batch = {
        "obs": torch.randn((16, 3), dtype=torch.float32),
        "actions": torch.rand((16, 1), dtype=torch.float32) * 2 - 1,
        "rewards": torch.randn(16, dtype=torch.float32),
        "next_obs": torch.randn((16, 3), dtype=torch.float32),
        "dones": torch.zeros(16, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=16)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy_term",
        "policy_delay",
    }
