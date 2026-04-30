import torch

from axiomrl.algorithms.openai_es import OpenAIES, openai_es_loss
from axiomrl.models.mlp_ars import MLPARSModel


def test_openai_es_loss_returns_named_metrics() -> None:
    metrics = openai_es_loss(
        {
            "positive_returns": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "negative_returns": torch.tensor([0.1, -0.2], dtype=torch.float32),
            "utilities": torch.tensor([0.3, -0.3], dtype=torch.float32),
            "reward_std": torch.tensor(0.8, dtype=torch.float32),
            "update": torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32),
            "parameters": torch.tensor([0.4, -0.3, 0.2], dtype=torch.float32),
        }
    )

    assert set(metrics) >= {
        "reward_std",
        "utility_mean",
        "update_norm",
        "parameter_norm",
        "positive_return_mean",
        "negative_return_mean",
    }


def test_openai_es_update_returns_update_result() -> None:
    torch.manual_seed(421)

    model = MLPARSModel(obs_dim=3, action_dim=1, hidden_sizes=(32, 32))
    algorithm = OpenAIES(
        model=model,
        step_size=0.02,
        noise_std=0.03,
    )

    perturbations = torch.randn((4, model.num_parameters), dtype=torch.float32)
    initial_params = model.flat_parameters().detach().clone()
    result = algorithm.update(
        {
            "perturbations": perturbations,
            "positive_returns": torch.tensor([1.2, 0.8, -0.1, 0.4], dtype=torch.float32),
            "negative_returns": torch.tensor([0.2, 0.1, -0.4, -0.3], dtype=torch.float32),
        },
        global_step=8,
    )

    assert result.num_gradient_steps == 1
    assert set(result.metrics) >= {
        "reward_std",
        "utility_mean",
        "update_norm",
        "parameter_norm",
        "positive_return_mean",
        "negative_return_mean",
    }
    assert not torch.allclose(model.flat_parameters(), initial_params)
