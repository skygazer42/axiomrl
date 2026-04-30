import torch

from axiomrl.algorithms.naf import NAF, naf_loss
from axiomrl.models.mlp_naf import MLPNAFModel


def test_mlp_naf_model_outputs_bounded_actions_and_scalar_values() -> None:
    model = MLPNAFModel(obs_dim=3, action_dim=2, hidden_sizes=(32, 32))
    obs = torch.zeros((4, 3), dtype=torch.float32)
    actions = torch.zeros((4, 2), dtype=torch.float32)

    greedy_actions = model.actor(obs)
    q_values = model.q_values(obs, actions)
    state_values = model.state_values(obs)

    assert greedy_actions.shape == (4, 2)
    assert q_values.shape == (4,)
    assert state_values.shape == (4,)
    assert torch.all(greedy_actions <= 1.0 + 1e-6)
    assert torch.all(greedy_actions >= -1.0 - 1e-6)


def test_naf_loss_returns_named_metrics() -> None:
    batch = {
        "q_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
    }

    metrics = naf_loss(batch)

    assert set(metrics) >= {"loss", "q_value_mean", "target_q_mean"}


def test_naf_update_returns_update_result() -> None:
    torch.manual_seed(211)

    model = MLPNAFModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = NAF(
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
    assert set(result.metrics) >= {"loss", "q_value_mean", "target_q_mean"}
