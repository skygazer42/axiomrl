import torch

from axiomrl.algorithms.a2c import A2C, a2c_loss
from axiomrl.models.mlp_actor_critic import MLPActorCritic


def test_a2c_loss_returns_named_metrics() -> None:
    batch = {
        "logprobs": torch.zeros(8),
        "advantages": torch.ones(8),
        "returns": torch.ones(8),
        "values": torch.zeros(8),
        "entropy": torch.ones(8),
    }

    metrics = a2c_loss(batch, ent_coef=0.01, vf_coef=0.5)

    assert set(metrics) >= {"loss", "policy_loss", "value_loss", "entropy_loss"}


def test_a2c_update_returns_update_result() -> None:
    torch.manual_seed(101)

    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = A2C(policy=policy, learning_rate=3e-4, ent_coef=0.01, vf_coef=0.5)

    obs = torch.randn((8, 4), dtype=torch.float32)
    rollout = policy.act(obs)
    batch = {
        "obs": obs,
        "actions": rollout.actions,
        "advantages": torch.ones(8, dtype=torch.float32),
        "returns": torch.ones(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"loss", "policy_loss", "value_loss", "entropy_loss"}
