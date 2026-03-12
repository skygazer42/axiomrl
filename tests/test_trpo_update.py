import pytest
import torch

from rl_training.algorithms.trpo import TRPO, trpo_loss
from rl_training.models.mlp_actor_critic import MLPActorCritic


def test_trpo_loss_returns_named_metrics() -> None:
    batch = {
        "logprobs": torch.zeros(8, dtype=torch.float32),
        "new_logprobs": torch.linspace(-0.1, 0.1, 8, dtype=torch.float32),
        "advantages": torch.linspace(-1.0, 1.0, 8, dtype=torch.float32),
        "returns": torch.ones(8, dtype=torch.float32),
        "values": torch.zeros(8, dtype=torch.float32),
        "entropy": torch.ones(8, dtype=torch.float32),
        "approx_kl": torch.as_tensor(0.005, dtype=torch.float32),
        "accepted_step": torch.as_tensor(1.0, dtype=torch.float32),
        "step_norm": torch.as_tensor(0.1, dtype=torch.float32),
        "backtrack_steps": torch.as_tensor(2.0, dtype=torch.float32),
        "cg_iterations": torch.as_tensor(5.0, dtype=torch.float32),
    }

    metrics = trpo_loss(batch)

    assert set(metrics) >= {
        "surrogate_gain",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "accepted_step",
        "step_norm",
        "backtrack_steps",
        "cg_iterations",
    }


def test_trpo_rejects_invalid_hyperparameters() -> None:
    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    with pytest.raises(ValueError, match="value_learning_rate must be > 0"):
        TRPO(policy=policy, value_learning_rate=0.0, max_kl=0.01)

    with pytest.raises(ValueError, match="max_kl must be > 0"):
        TRPO(policy=policy, value_learning_rate=1e-3, max_kl=0.0)

    with pytest.raises(ValueError, match="cg_iterations must be >= 1"):
        TRPO(policy=policy, value_learning_rate=1e-3, max_kl=0.01, cg_iterations=0)

    with pytest.raises(ValueError, match="line_search_shrink must be in \\(0, 1\\)"):
        TRPO(policy=policy, value_learning_rate=1e-3, max_kl=0.01, line_search_shrink=1.0)

    with pytest.raises(ValueError, match="value_updates must be >= 1"):
        TRPO(policy=policy, value_learning_rate=1e-3, max_kl=0.01, value_updates=0)


def test_trpo_update_returns_update_result() -> None:
    torch.manual_seed(29)

    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = TRPO(
        policy=policy,
        value_learning_rate=1e-3,
        max_kl=0.01,
        cg_iterations=5,
        cg_damping=0.1,
        line_search_steps=5,
        line_search_shrink=0.8,
        value_updates=3,
    )

    obs = torch.randn((8, 4), dtype=torch.float32)
    rollout = policy.act(obs)
    batch = {
        "obs": obs,
        "actions": rollout.actions,
        "logprobs": rollout.logprobs.detach(),
        "advantages": torch.linspace(-1.0, 1.0, 8, dtype=torch.float32),
        "returns": torch.linspace(0.0, 1.0, 8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 4
    assert set(result.metrics) >= {
        "surrogate_gain",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "accepted_step",
        "step_norm",
        "backtrack_steps",
        "cg_iterations",
        "line_search_fraction",
        "value_updates",
        "final_value_loss",
        "surrogate_improvement",
    }
