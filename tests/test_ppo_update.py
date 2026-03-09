import torch

from rl_training.algorithms.ppo import PPO, ppo_loss
from rl_training.models.mlp_actor_critic import MLPActorCritic


def test_mlp_actor_critic_act_returns_policy_output() -> None:
    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    output = policy.act(torch.zeros((3, 4), dtype=torch.float32))

    assert output.actions.shape == (3,)
    assert output.logprobs.shape == (3,)
    assert output.values.shape == (3,)
    assert output.entropy.shape == (3,)


def test_ppo_loss_returns_named_metrics() -> None:
    minibatch = {
        "logprobs": torch.zeros(8),
        "new_logprobs": torch.zeros(8),
        "advantages": torch.ones(8),
        "returns": torch.ones(8),
        "values": torch.zeros(8),
        "new_values": torch.zeros(8),
        "entropy": torch.ones(8),
    }

    metrics = ppo_loss(minibatch, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5)

    assert set(metrics) >= {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    }


def test_ppo_update_returns_update_result() -> None:
    torch.manual_seed(7)

    policy = MLPActorCritic(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = PPO(policy=policy, learning_rate=3e-4, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5)

    obs = torch.randn((8, 4), dtype=torch.float32)
    rollout = policy.act(obs)
    batch = {
        "obs": obs,
        "actions": rollout.actions,
        "logprobs": rollout.logprobs.detach(),
        "advantages": torch.ones(8, dtype=torch.float32),
        "returns": torch.ones(8, dtype=torch.float32),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"loss", "policy_loss", "value_loss", "entropy_loss"}
