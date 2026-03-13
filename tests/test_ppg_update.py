import copy

import torch

from rl_training.algorithms.ppg import PPG, ppg_auxiliary_loss, ppg_loss
from rl_training.models import MLPPPGModel


def test_mlp_ppg_model_act_returns_policy_output() -> None:
    model = MLPPPGModel(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))

    output = model.act(torch.zeros((3, 4), dtype=torch.float32))

    assert output.actions.shape == (3,)
    assert output.logprobs.shape == (3,)
    assert output.values.shape == (3,)
    assert output.entropy.shape == (3,)


def test_ppg_loss_returns_named_metrics() -> None:
    minibatch = {
        "logprobs": torch.zeros(8),
        "new_logprobs": torch.zeros(8),
        "advantages": torch.ones(8),
        "returns": torch.ones(8),
        "new_values": torch.zeros(8),
        "entropy": torch.ones(8),
    }

    metrics = ppg_loss(minibatch, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5)

    assert set(metrics) >= {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    }


def test_ppg_auxiliary_loss_returns_named_metrics() -> None:
    batch = {
        "policy_logits": torch.zeros((8, 2), dtype=torch.float32),
        "teacher_logits": torch.zeros((8, 2), dtype=torch.float32),
        "values": torch.zeros(8, dtype=torch.float32),
        "auxiliary_values": torch.zeros(8, dtype=torch.float32),
        "returns": torch.ones(8, dtype=torch.float32),
    }

    metrics = ppg_auxiliary_loss(batch, aux_value_coef=1.0, behavior_clone_coef=1.0, value_clone_coef=1.0)

    assert set(metrics) >= {"auxiliary_loss", "aux_value_loss", "behavior_clone_loss", "value_clone_loss"}


def test_ppg_update_and_auxiliary_update_return_metrics() -> None:
    torch.manual_seed(7)

    model = MLPPPGModel(obs_dim=4, action_dim=2, hidden_sizes=(32, 32))
    algorithm = PPG(
        model=model,
        learning_rate=3e-4,
        aux_learning_rate=3e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        aux_value_coef=1.0,
        behavior_clone_coef=1.0,
        value_clone_coef=1.0,
    )

    obs = torch.randn((8, 4), dtype=torch.float32)
    rollout = model.act(obs)
    batch = {
        "obs": obs,
        "actions": rollout.actions,
        "logprobs": rollout.logprobs.detach(),
        "advantages": torch.ones(8, dtype=torch.float32),
        "returns": torch.ones(8, dtype=torch.float32),
    }

    policy_result = algorithm.update(batch, global_step=8)
    teacher_model = copy.deepcopy(algorithm.model)
    auxiliary_result = algorithm.auxiliary_update(
        {
            "obs": obs,
            "returns": torch.ones(8, dtype=torch.float32),
        },
        teacher_model=teacher_model,
        global_step=8,
    )

    assert policy_result.num_gradient_steps == 1
    assert set(policy_result.metrics) >= {"loss", "policy_loss", "value_loss", "entropy_loss", "policy_updates"}
    assert auxiliary_result.num_gradient_steps == 1
    assert set(auxiliary_result.metrics) >= {
        "auxiliary_loss",
        "aux_value_loss",
        "behavior_clone_loss",
        "value_clone_loss",
        "auxiliary_updates",
    }
