import pytest
import torch
from torch import nn

from axiomrl.algorithms.pets import PETS, pets_loss
from axiomrl.models.mlp_mopo import MLPMOPOEnsembleModel


class ToyPETSModel(nn.Module):
    def __init__(self, *, target_action: float = 0.35) -> None:
        super().__init__()
        self.target_action = float(target_action)
        self.obs_dim = 2
        self.action_dim = 1
        self.num_ensembles = 3
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def ensemble_parameters(self):
        yield self.bias

    def predict_distribution(self, obs: object, actions: object) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=obs_tensor.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(-1)

        reward = -((action_tensor[:, 0] - self.target_action) ** 2) + self.bias.to(obs_tensor.device)
        delta_obs = torch.zeros(
            (obs_tensor.shape[0], self.obs_dim),
            dtype=torch.float32,
            device=obs_tensor.device,
        )
        means = torch.cat([delta_obs, reward.unsqueeze(-1)], dim=-1).unsqueeze(0).repeat(self.num_ensembles, 1, 1)
        logvars = torch.full_like(means, -10.0)
        return means, logvars


def test_pets_loss_returns_named_metrics() -> None:
    metrics = pets_loss(
        {
            "predicted_means": torch.zeros((3, 2, 4), dtype=torch.float32),
            "predicted_logvars": torch.zeros((3, 2, 4), dtype=torch.float32),
            "targets": torch.ones((2, 4), dtype=torch.float32),
            "ensemble_disagreement": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }
    )

    assert set(metrics) >= {
        "pets_model_loss",
        "reward_mae",
        "delta_obs_mae",
        "ensemble_disagreement",
    }


def test_pets_planner_moves_toward_high_reward_action() -> None:
    torch.manual_seed(123)

    algorithm = PETS(
        dynamics_model=ToyPETSModel(target_action=0.35),
        learning_rate=1e-3,
    )

    action = algorithm.plan_action(
        [0.0, 0.0],
        action_low=[-1.0],
        action_high=[1.0],
        horizon=3,
        num_candidates=128,
        num_iterations=4,
        num_topk=16,
        num_particles=1,
        deterministic=True,
    )

    assert action.shape == (1,)
    assert action[0] == pytest.approx(0.35, abs=0.2)


def test_pets_update_returns_update_result() -> None:
    torch.manual_seed(321)

    model = MLPMOPOEnsembleModel(
        obs_dim=3,
        action_dim=1,
        hidden_sizes=(32, 32),
        num_ensembles=3,
    )
    algorithm = PETS(
        dynamics_model=model,
        learning_rate=1e-3,
    )

    obs = torch.randn((64, 3), dtype=torch.float32)
    actions = torch.tanh(torch.randn((64, 1), dtype=torch.float32))
    next_obs = obs + 0.1 * actions.repeat(1, 3)
    rewards = 1.0 - actions[:, 0].abs()
    initial_parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()

    result = algorithm.update(
        {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
        },
        global_step=64,
    )

    updated_parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "pets_model_loss",
        "reward_mae",
        "delta_obs_mae",
        "ensemble_disagreement",
    }
    assert not torch.allclose(updated_parameters, initial_parameters)
