import torch

from rl_training.algorithms.mopo import MOPO, mopo_model_loss
from rl_training.models import MLPMOPOEnsembleModel, MLPSACModel


def test_mopo_model_loss_returns_named_metrics() -> None:
    metrics = mopo_model_loss(
        {
            "predicted_means": torch.zeros((3, 8, 4), dtype=torch.float32),
            "predicted_logvars": torch.zeros((3, 8, 4), dtype=torch.float32),
            "targets": torch.ones((8, 4), dtype=torch.float32),
            "ensemble_disagreement": torch.ones(8, dtype=torch.float32),
        }
    )

    assert set(metrics) == {"mopo_model_loss", "reward_mae", "delta_obs_mae", "ensemble_disagreement"}


def test_mopo_update_model_and_policy_return_expected_metrics() -> None:
    torch.manual_seed(211)

    algorithm = MOPO(
        policy_model=MLPSACModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32)),
        dynamics_model=MLPMOPOEnsembleModel(
            obs_dim=3,
            action_dim=1,
            hidden_sizes=(32, 32),
            num_ensembles=3,
        ),
        policy_learning_rate=3e-4,
        model_learning_rate=1e-3,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
        penalty_coef=1.0,
    )

    model_result = algorithm.update_model(
        {
            "obs": torch.randn((8, 3), dtype=torch.float32),
            "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
            "rewards": torch.randn(8, dtype=torch.float32),
            "next_obs": torch.randn((8, 3), dtype=torch.float32),
        },
        global_step=0,
    )
    policy_result = algorithm.update(
        {
            "obs": torch.randn((8, 3), dtype=torch.float32),
            "actions": torch.rand((8, 1), dtype=torch.float32) * 2 - 1,
            "rewards": torch.randn(8, dtype=torch.float32),
            "next_obs": torch.randn((8, 3), dtype=torch.float32),
            "dones": torch.zeros(8, dtype=torch.float32),
        },
        global_step=8,
    )

    assert model_result.num_gradient_steps == 1
    assert set(model_result.metrics) >= {"mopo_model_loss", "reward_mae", "delta_obs_mae", "ensemble_disagreement"}
    assert policy_result.num_gradient_steps == 1
    assert set(policy_result.metrics) >= {"critic_loss", "actor_loss", "target_q_mean", "entropy_term"}
