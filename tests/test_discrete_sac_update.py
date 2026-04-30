import torch

from axiomrl.algorithms.discrete_sac import DiscreteSAC, discrete_sac_loss
from axiomrl.models.mlp_discrete_sac import MLPDiscreteSACModel


def test_discrete_sac_loss_returns_named_metrics() -> None:
    batch = {
        "q1_values": torch.zeros(8, dtype=torch.float32),
        "q2_values": torch.zeros(8, dtype=torch.float32),
        "target_q_values": torch.ones(8, dtype=torch.float32),
        "action_probs": torch.full((8, 2), 0.5, dtype=torch.float32),
        "log_action_probs": torch.log(torch.full((8, 2), 0.5, dtype=torch.float32)),
        "policy_q1": torch.zeros((8, 2), dtype=torch.float32),
        "policy_q2": torch.zeros((8, 2), dtype=torch.float32),
        "alpha": torch.as_tensor(0.2, dtype=torch.float32),
    }

    metrics = discrete_sac_loss(batch)

    assert set(metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy",
    }


def test_discrete_sac_update_returns_update_result() -> None:
    torch.manual_seed(31)

    model = MLPDiscreteSACModel(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = DiscreteSAC(
        model=model,
        learning_rate=3e-4,
        gamma=0.99,
        alpha=0.2,
        tau=0.005,
    )

    batch = {
        "obs": torch.randn((16, 4), dtype=torch.float32),
        "actions": torch.randint(0, 2, (16,), dtype=torch.int64),
        "rewards": torch.randn(16, dtype=torch.float32),
        "next_obs": torch.randn((16, 4), dtype=torch.float32),
        "dones": torch.zeros(16, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=16)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "critic_loss",
        "actor_loss",
        "target_q_mean",
        "entropy",
    }
