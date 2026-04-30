import torch

from axiomrl.algorithms.tqc import TQC, _truncate_quantiles, tqc_loss
from axiomrl.models.mlp_tqc import MLPTQCModel


def test_mlp_tqc_model_samples_bounded_actions_and_quantiles() -> None:
    model = MLPTQCModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_critics=3,
        num_quantiles=7,
    )

    sampled = model.sample_actions(torch.zeros((4, 3), dtype=torch.float32))
    quantiles = model.quantile_values(
        torch.zeros((4, 3), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32)
    )

    assert sampled.actions.shape == (4, 1)
    assert sampled.logprobs.shape == (4,)
    assert quantiles.shape == (4, 3, 7)
    assert torch.all(sampled.actions <= 1.0 + 1e-6)
    assert torch.all(sampled.actions >= -1.0 - 1e-6)


def test_truncate_quantiles_drops_top_values_across_all_critics() -> None:
    quantiles = torch.tensor(
        [
            [
                [1.0, 5.0, 2.0],
                [3.0, 6.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )

    truncated = _truncate_quantiles(quantiles, top_quantiles_to_drop_per_net=1)

    assert truncated.shape == (1, 4)
    assert truncated.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_tqc_loss_returns_named_metrics() -> None:
    batch = {
        "critic_quantiles": torch.zeros((8, 2, 5), dtype=torch.float32),
        "target_quantiles": torch.ones((8, 6), dtype=torch.float32),
        "taus": torch.full((5,), 0.5, dtype=torch.float32),
        "sampled_logprobs": torch.zeros(8, dtype=torch.float32),
        "sampled_q_values": torch.zeros(8, dtype=torch.float32),
        "alpha": 0.2,
        "kappa": 1.0,
    }

    metrics = tqc_loss(batch)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}


def test_tqc_update_returns_update_result() -> None:
    torch.manual_seed(19)

    model = MLPTQCModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_critics=3,
        num_quantiles=7,
    )
    algorithm = TQC(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        top_quantiles_to_drop_per_net=1,
        num_quantiles=7,
        kappa=1.0,
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
