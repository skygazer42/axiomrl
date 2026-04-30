import torch

from axiomrl.algorithms.ddpg import DDPG, ddpg_loss
from axiomrl.models.mlp_ddpg import MLPDDPGModel


def test_mlp_ddpg_model_actor_outputs_bounded_actions() -> None:
    model = MLPDDPGModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    actions = model.actor(torch.zeros((4, 3), dtype=torch.float32))

    assert actions.shape == (4, 1)
    assert torch.all(actions <= 1.0 + 1e-6)
    assert torch.all(actions >= -1.0 - 1e-6)


def test_ddpg_loss_returns_named_metrics() -> None:
    batch = {
        "q_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "actor_q_values": torch.zeros(8, dtype=torch.float32),
    }

    metrics = ddpg_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean"}


def test_ddpg_update_returns_update_result() -> None:
    torch.manual_seed(151)

    model = MLPDDPGModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = DDPG(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
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
