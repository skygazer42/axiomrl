import torch

from axiomrl.algorithms.sac import SAC, sac_loss
from axiomrl.models.mlp_sac import MLPSACModel


def test_mlp_sac_model_samples_bounded_actions() -> None:
    model = MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))

    output = model.sample_actions(torch.zeros((4, 3), dtype=torch.float32))

    assert output.actions.shape == (4, 1)
    assert output.logprobs.shape == (4,)
    assert torch.all(output.actions <= 1.0 + 1e-6)
    assert torch.all(output.actions >= -1.0 - 1e-6)


def test_sac_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q1": torch.zeros(8, dtype=torch.float32),
        "sampled_q2": torch.zeros(8, dtype=torch.float32),
        "alpha": 0.2,
    }

    metrics = sac_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}


def test_sac_update_returns_update_result() -> None:
    torch.manual_seed(19)

    model = MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = SAC(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
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
    assert set(result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}
