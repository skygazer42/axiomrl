import torch

from rl_training.algorithms.decision_transformer import DecisionTransformer, decision_transformer_loss
from rl_training.models import DecisionTransformerModel


def test_decision_transformer_model_predict_actions_returns_expected_shape() -> None:
    model = DecisionTransformerModel(
        obs_dim=3,
        action_dim=2,
        context_length=4,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_timestep=16,
    )

    actions = model.predict_actions(
        obs=torch.randn(5, 4, 3),
        actions=torch.randn(5, 4, 2),
        returns_to_go=torch.randn(5, 4),
        timesteps=torch.tensor([[0, 1, 2, 3]] * 5, dtype=torch.int64),
        mask=torch.ones(5, 4),
    )

    assert actions.shape == (5, 4, 2)


def test_decision_transformer_loss_returns_named_metrics() -> None:
    metrics = decision_transformer_loss(
        {
            "predictions": torch.zeros((2, 4, 2), dtype=torch.float32),
            "targets": torch.ones((2, 4, 2), dtype=torch.float32),
            "mask": torch.tensor([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        }
    )

    assert set(metrics) == {"decision_transformer_loss", "action_mse", "masked_tokens"}


def test_decision_transformer_update_returns_expected_metrics() -> None:
    model = DecisionTransformerModel(
        obs_dim=3,
        action_dim=2,
        context_length=4,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_timestep=16,
    )
    algorithm = DecisionTransformer(
        model=model,
        learning_rate=1e-4,
    )
    batch = {
        "obs": torch.randn(8, 4, 3),
        "actions": torch.randn(8, 4, 2).clamp(-1.0, 1.0),
        "returns_to_go": torch.randn(8, 4),
        "timesteps": torch.tensor([[0, 1, 2, 3]] * 8, dtype=torch.int64),
        "mask": torch.ones(8, 4),
    }

    result = algorithm.update(batch, global_step=8)

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {"decision_transformer_loss", "action_mse", "masked_tokens"}
