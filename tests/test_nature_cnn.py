import pytest
import torch

from axiomrl.models.cnn import NatureCNN


def test_nature_cnn_returns_fixed_size_features_for_batched_images() -> None:
    encoder = NatureCNN(obs_shape=(4, 84, 84), features_dim=256)

    obs = torch.zeros((3, 4, 84, 84), dtype=torch.uint8)
    features = encoder(obs)

    assert features.shape == (3, 256)
    assert encoder.features_dim == 256


def test_nature_cnn_accepts_single_image_observation() -> None:
    encoder = NatureCNN(obs_shape=(4, 84, 84), features_dim=128)

    obs = torch.zeros((4, 84, 84), dtype=torch.uint8)
    features = encoder(obs)

    assert features.shape == (1, 128)


def test_nature_cnn_rejects_non_image_observation_shapes() -> None:
    with pytest.raises(ValueError, match="expects 3D channel-first observations"):
        NatureCNN(obs_shape=(84, 84), features_dim=128)
