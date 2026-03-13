import torch

from rl_training.algorithms.d4pg import D4PG, d4pg_loss
from rl_training.models.mlp_d4pg import MLPD4PGModel


def test_mlp_d4pg_model_outputs_bounded_actions_and_distribution_logits() -> None:
    model = MLPD4PGModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        v_min=-10.0,
        v_max=10.0,
        num_atoms=21,
    )
    obs = torch.zeros((4, 3), dtype=torch.float32)
    actions = torch.zeros((4, 1), dtype=torch.float32)

    actor_actions = model.actor(obs)
    logits = model.distribution_logits(obs, actions)
    q_values = model.q_values(obs, actions)

    assert actor_actions.shape == (4, 1)
    assert logits.shape == (4, 21)
    assert q_values.shape == (4,)
    assert torch.all(actor_actions <= 1.0 + 1e-6)
    assert torch.all(actor_actions >= -1.0 - 1e-6)


def test_d4pg_loss_returns_named_metrics() -> None:
    batch = {
        "logits": torch.zeros((8, 21), dtype=torch.float32),
        "target_distributions": torch.ones((8, 21), dtype=torch.float32) / 21.0,
        "actor_q_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
    }

    metrics = d4pg_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean"}


def test_d4pg_update_returns_update_result() -> None:
    torch.manual_seed(307)

    model = MLPD4PGModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        v_min=-10.0,
        v_max=10.0,
        num_atoms=21,
    )
    algorithm = D4PG(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        v_min=-10.0,
        v_max=10.0,
        num_atoms=21,
    )

    batch = {
        "obs": torch.randn((8, 3), dtype=torch.float32),
        "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
        "rewards": torch.randn(8, dtype=torch.float32),
        "next_obs": torch.randn((8, 3), dtype=torch.float32),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean"}
