import torch

from axiomrl.algorithms.fqf import FQF
from axiomrl.models.mlp_fqf_network import MLPFQFNetwork


def test_mlp_fqf_network_forward_shapes_and_tau_bounds() -> None:
    torch.manual_seed(0)

    network = MLPFQFNetwork(
        obs_dim=4,
        action_dim=2,
        num_quantiles=8,
        hidden_sizes=(32, 32),
        embedding_dim=16,
    )

    out = network(torch.zeros((5, 4), dtype=torch.float32))

    assert out.quantile_hats.shape == (5, 2, 8)
    assert out.taus.shape == (5, 9)
    assert out.tau_hats.shape == (5, 8)
    assert out.quantiles_tau.shape == (5, 2, 7)

    assert torch.all(out.taus >= 0.0)
    assert torch.all(out.taus <= 1.0)
    assert torch.all(out.tau_hats >= 0.0)
    assert torch.all(out.tau_hats <= 1.0)
    assert torch.all(out.taus[:, 1:] >= out.taus[:, :-1])

    assert torch.allclose(out.taus[:, :1], torch.zeros((5, 1), dtype=torch.float32), atol=1e-5)
    assert torch.allclose(out.taus[:, -1:], torch.ones((5, 1), dtype=torch.float32), atol=1e-5)


def test_fqf_update_returns_update_result() -> None:
    torch.manual_seed(7)

    network = MLPFQFNetwork(
        obs_dim=4,
        action_dim=2,
        num_quantiles=8,
        hidden_sizes=(32, 32),
        embedding_dim=16,
    )
    algorithm = FQF(
        q_network=network,
        learning_rate=1e-3,
        fraction_learning_rate=5e-4,
        gamma=0.99,
        target_update_interval=2,
        num_quantiles=8,
        kappa=1.0,
        entropy_coef=1e-3,
    )

    batch = {
        "obs": torch.randn((8, 4), dtype=torch.float32),
        "actions": torch.randint(0, 2, (8,), dtype=torch.int64),
        "rewards": torch.randn(8, dtype=torch.float32),
        "next_obs": torch.randn((8, 4), dtype=torch.float32),
        "dones": torch.zeros(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "loss",
        "fraction_loss",
        "entropy_loss",
        "q_value_mean",
        "target_mean",
        "td_error_mean",
    }
