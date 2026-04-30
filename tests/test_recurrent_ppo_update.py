import torch

from axiomrl.contrib.recurrent_ppo import RecurrentPPOAlgorithm
from axiomrl.models import LSTMActorCritic


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "obs": torch.zeros((4, 2, 4), dtype=torch.float32),
        "actions": torch.zeros((4, 2), dtype=torch.int64),
        "logprobs": torch.zeros((4, 2), dtype=torch.float32),
        "advantages": torch.ones((4, 2), dtype=torch.float32),
        "returns": torch.zeros((4, 2), dtype=torch.float32),
        "episode_starts": torch.zeros((4, 2), dtype=torch.float32),
        "mask": torch.ones((4, 2), dtype=torch.float32),
        "initial_h": torch.zeros((1, 2, 32), dtype=torch.float32),
        "initial_c": torch.zeros((1, 2, 32), dtype=torch.float32),
    }


def test_recurrent_ppo_update_returns_named_metrics() -> None:
    algorithm = RecurrentPPOAlgorithm(
        policy=LSTMActorCritic(
            obs_shape=(4,),
            action_dim=2,
            features_dim=32,
            encoder_hidden_sizes=(16,),
            head_hidden_sizes=(16,),
            hidden_size=32,
            num_layers=1,
        ),
        learning_rate=3e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    result = algorithm.update(_make_batch(), global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    }


def test_recurrent_ppo_update_supports_masked_sequences() -> None:
    batch = _make_batch()
    batch["mask"][-1] = 0.0

    algorithm = RecurrentPPOAlgorithm(
        policy=LSTMActorCritic(
            obs_shape=(4,),
            action_dim=2,
            features_dim=32,
            encoder_hidden_sizes=(16,),
            head_hidden_sizes=(16,),
            hidden_size=32,
            num_layers=1,
        ),
        learning_rate=3e-4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    result = algorithm.update(batch, global_step=16)

    assert "loss" in result.metrics
