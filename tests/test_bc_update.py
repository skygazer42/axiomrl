import torch

from rl_training.algorithms.bc import BC
from rl_training.models import MLPBCModel


def test_bc_update_returns_named_metrics() -> None:
    algorithm = BC(
        model=MLPBCModel(obs_dim=3, action_dim=1, hidden_sizes=(16, 16)),
        learning_rate=3e-4,
    )

    result = algorithm.update(
        {
            "obs": torch.zeros((8, 3), dtype=torch.float32),
            "actions": torch.zeros((8, 1), dtype=torch.float32),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"bc_loss", "action_abs_mean"}
