import torch

from axiomrl.models import MLPMOPOEnsembleModel


def test_mopo_dynamics_model_predicts_distribution_and_samples_transitions() -> None:
    model = MLPMOPOEnsembleModel(
        obs_dim=3,
        action_dim=2,
        hidden_sizes=(32, 32),
        num_ensembles=5,
    )

    obs = torch.randn(8, 3)
    actions = torch.randn(8, 2).clamp(-1.0, 1.0)
    means, logvars = model.predict_distribution(obs, actions)
    sample = model.sample_transition(obs, actions)

    assert means.shape == (5, 8, 4)
    assert logvars.shape == (5, 8, 4)
    assert sample["next_obs"].shape == (8, 3)
    assert sample["rewards"].shape == (8,)
    assert sample["disagreement"].shape == (8,)
