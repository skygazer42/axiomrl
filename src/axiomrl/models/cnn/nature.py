from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class NatureCNN(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        features_dim: int = 512,
        normalize_images: bool = True,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if len(self.obs_shape) != 3:
            raise ValueError(f"NatureCNN expects 3D channel-first observations, got {self.obs_shape!r}")

        self.normalize_images = normalize_images
        channels = self.obs_shape[0]
        self.convs = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros((1, *self.obs_shape), dtype=torch.float32)
            n_flatten = int(self.convs(sample).shape[1])

        self.projection = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        if self.normalize_images:
            obs_tensor = obs_tensor / 255.0
        features = self.convs(obs_tensor)
        return self.projection(features)
