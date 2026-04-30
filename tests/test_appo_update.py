import torch

from axiomrl.algorithms.appo import APPO, appo_loss
from axiomrl.models.mlp_actor_critic import MLPActorCritic


def test_appo_loss_returns_named_metrics() -> None:
    metrics = appo_loss(
        {
            "target_logprobs": torch.zeros((4, 2), dtype=torch.float32),
            "behavior_logprobs": torch.full((4, 2), -0.1, dtype=torch.float32),
            "vtrace_targets": torch.ones((4, 2), dtype=torch.float32),
            "values": torch.zeros((4, 2), dtype=torch.float32),
            "pg_advantages": torch.ones((4, 2), dtype=torch.float32),
            "entropy": torch.ones((4, 2), dtype=torch.float32),
        },
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    assert set(metrics) >= {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
        "rho_mean",
        "vtrace_target_mean",
        "pg_advantage_mean",
    }


def test_appo_update_returns_expected_metrics() -> None:
    torch.manual_seed(313)

    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = APPO(
        policy=policy,
        learning_rate=3e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gamma=0.99,
        rho_clip=1.0,
        c_clip=1.0,
        pg_rho_clip=1.0,
    )

    obs = torch.randn((5, 3, 4), dtype=torch.float32)
    flat_obs = obs.reshape(-1, 4)
    rollout = policy.act(flat_obs)
    actions = rollout.actions.reshape(5, 3)
    behavior_logprobs = rollout.logprobs.detach().reshape(5, 3)

    result = algorithm.update(
        {
            "obs": obs,
            "actions": actions,
            "rewards": torch.randn((5, 3), dtype=torch.float32),
            "dones": torch.zeros((5, 3), dtype=torch.float32),
            "behavior_logprobs": behavior_logprobs,
            "bootstrap_value": torch.zeros(3, dtype=torch.float32),
        },
        global_step=15,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
        "rho_mean",
        "vtrace_target_mean",
        "pg_advantage_mean",
    }
