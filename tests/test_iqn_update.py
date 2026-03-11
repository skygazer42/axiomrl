import torch

from rl_training.algorithms.iqn import IQN
from rl_training.models.mlp_iqn_network import MLPIQNetwork


def test_mlp_iqn_network_forward_shape_and_tau_range() -> None:
    network = MLPIQNetwork(
        obs_dim=4,
        action_dim=2,
        num_quantiles=8,
        hidden_sizes=(32, 32),
        embedding_dim=16,
    )

    quantiles, taus = network(torch.zeros((5, 4), dtype=torch.float32))

    assert quantiles.shape == (5, 2, 8)
    assert taus.shape == (5, 8)
    assert torch.all(taus >= 0.0)
    assert torch.all(taus <= 1.0)


def test_mlp_iqn_network_act_uses_mean_over_quantiles() -> None:
    class FixedIQNetwork(MLPIQNetwork):
        def __init__(self) -> None:
            super().__init__(
                obs_dim=1,
                action_dim=2,
                num_quantiles=3,
                hidden_sizes=(8,),
                embedding_dim=8,
            )

        def forward(
            self,
            obs: torch.Tensor,
            *,
            num_quantiles: int | None = None,
            taus: torch.Tensor | None = None,
            random_taus: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
            del num_quantiles, taus, random_taus
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            batch_size = int(obs_tensor.shape[0])
            values = torch.tensor(
                [[[0.5, 0.5, 0.5], [100.0, -100.0, 0.0]]],
                dtype=torch.float32,
            )
            taus = torch.full((batch_size, 3), 0.5, dtype=torch.float32)
            return values.repeat(batch_size, 1, 1), taus

    network = FixedIQNetwork()

    action = network.act(torch.zeros((1, 1), dtype=torch.float32), epsilon=0.0)

    assert int(action.item()) == 0


def test_iqn_update_returns_update_result() -> None:
    torch.manual_seed(7)

    network = MLPIQNetwork(
        obs_dim=4,
        action_dim=2,
        num_quantiles=8,
        hidden_sizes=(32, 32),
        embedding_dim=16,
    )
    algorithm = IQN(
        q_network=network,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_interval=2,
        num_quantiles=8,
        kappa=1.0,
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
    assert set(result.metrics) >= {"loss", "q_value_mean", "target_mean", "td_error_mean"}
