import torch

from rl_training.models.mlp_mopo import MLPMOPOEnsembleModel
from rl_training.models.mlp_sac import MLPSACModel


def test_mbpo_update_and_model_update_return_metrics() -> None:
    from rl_training.algorithms.mbpo import MBPO

    torch.manual_seed(0)

    obs_dim = 3
    action_dim = 1
    batch_size = 8

    policy_model = MLPSACModel(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(32, 32))
    dynamics_model = MLPMOPOEnsembleModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(32, 32),
        num_ensembles=3,
    )
    algorithm = MBPO(
        policy_model=policy_model,
        dynamics_model=dynamics_model,
        policy_learning_rate=1e-3,
        model_learning_rate=1e-3,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
    )

    batch = {
        "obs": torch.randn(batch_size, obs_dim),
        "actions": torch.tanh(torch.randn(batch_size, action_dim)),
        "rewards": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, obs_dim),
        "dones": torch.zeros(batch_size),
    }

    model_result = algorithm.update_model(batch, global_step=0)
    assert model_result.num_gradient_steps == 1
    assert "mbpo_model_loss" in model_result.metrics
    assert "reward_mae" in model_result.metrics

    policy_result = algorithm.update(batch, global_step=0)
    assert policy_result.num_gradient_steps == 1
    assert "actor_loss" in policy_result.metrics
    assert "critic_loss" in policy_result.metrics

