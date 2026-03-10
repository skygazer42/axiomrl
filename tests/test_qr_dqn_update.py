import torch

from rl_training.algorithms.qr_dqn import QRDQN
from rl_training.models.mlp_qr_q_network import MLPQRQNetwork


def test_mlp_qr_q_network_forward_shape() -> None:
    network = MLPQRQNetwork(obs_dim=4, action_dim=2, num_quantiles=51, hidden_sizes=(32, 32))

    quantiles = network(torch.zeros((5, 4), dtype=torch.float32))

    assert quantiles.shape == (5, 2, 51)


def test_mlp_qr_q_network_act_uses_mean_over_quantiles() -> None:
    class FixedQuantileNetwork(MLPQRQNetwork):
        def __init__(self) -> None:
            super().__init__(obs_dim=1, action_dim=2, num_quantiles=3, hidden_sizes=())

        def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            batch_size = int(obs_tensor.shape[0])
            # action 0 mean = 0.5, action 1 mean = 0.0 (but max quantile is huge).
            values = torch.tensor(
                [[[0.5, 0.5, 0.5], [100.0, -100.0, 0.0]]],
                dtype=torch.float32,
            )
            return values.repeat(batch_size, 1, 1)

    network = FixedQuantileNetwork()

    action = network.act(torch.zeros((1, 1), dtype=torch.float32), epsilon=0.0)

    assert int(action.item()) == 0


def test_qr_dqn_update_returns_update_result() -> None:
    torch.manual_seed(7)

    network = MLPQRQNetwork(obs_dim=4, action_dim=2, num_quantiles=51, hidden_sizes=(32, 32))
    algorithm = QRDQN(
        q_network=network,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_interval=2,
        num_quantiles=51,
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
