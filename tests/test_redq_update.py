import torch
import pytest

from rl_training.algorithms.redq import REDQ, _sample_target_critic_indices, redq_loss
from rl_training.models.mlp_redq import MLPREDQModel


def test_mlp_redq_model_samples_bounded_actions_and_ensemble_q_values() -> None:
    model = MLPREDQModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_critics=5,
    )

    sampled = model.sample_actions(torch.zeros((4, 3), dtype=torch.float32))
    q_values = model.q_values(torch.zeros((4, 3), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32))

    assert sampled.actions.shape == (4, 1)
    assert sampled.logprobs.shape == (4,)
    assert q_values.shape == (4, 5)
    assert torch.all(sampled.actions <= 1.0 + 1e-6)
    assert torch.all(sampled.actions >= -1.0 - 1e-6)


def test_mlp_redq_q_values_accepts_single_multidimensional_action() -> None:
    model = MLPREDQModel(
        obs_dim=4,
        action_dim=2,
        hidden_sizes=(32, 32),
        num_critics=5,
    )

    q_values = model.q_values(
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
    )

    assert q_values.shape == (1, 5)


def test_sample_target_critic_indices_returns_unique_subset() -> None:
    indices = _sample_target_critic_indices(num_critics=10, subset_size=3)

    assert indices.shape == (3,)
    assert len(set(indices.tolist())) == 3
    assert all(0 <= int(index) < 10 for index in indices.tolist())


def test_redq_rejects_invalid_subset_size_at_construction() -> None:
    model = MLPREDQModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_critics=5,
    )

    with pytest.raises(ValueError, match="subset_size must be >= 1"):
        REDQ(
            model=model,
            learning_rate=3e-4,
            gamma=0.99,
            alpha=0.2,
            tau=0.005,
            num_critics=5,
            subset_size=0,
        )


def test_redq_loss_returns_named_metrics() -> None:
    batch = {
        "critic_q_values": torch.zeros((8, 5), dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q_values": torch.zeros(8, dtype=torch.float32),
        "alpha": 0.2,
    }

    metrics = redq_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}


def test_redq_update_returns_update_result() -> None:
    torch.manual_seed(23)

    model = MLPREDQModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_critics=5,
    )
    algorithm = REDQ(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        num_critics=5,
        subset_size=2,
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
