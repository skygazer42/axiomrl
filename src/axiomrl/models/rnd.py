from collections.abc import Sequence

import torch
from torch import nn

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.mlp_q_network import _build_mlp


def _build_rnd_encoder(
    *,
    obs_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    embedding_dim: int,
    activation: type[nn.Module],
) -> nn.Module:
    shape = tuple(int(dim) for dim in obs_shape)
    if len(shape) == 1:
        return _build_mlp(
            input_dim=shape[0],
            hidden_sizes=hidden_sizes,
            output_dim=embedding_dim,
            activation=activation,
        )

    if len(shape) == 3:
        if hidden_sizes:
            conv_features_dim = int(hidden_sizes[0])
            head_hidden_sizes = tuple(int(size) for size in hidden_sizes[1:])
        else:
            conv_features_dim = int(embedding_dim)
            head_hidden_sizes = ()

        encoder_layers: list[nn.Module] = [NatureCNN(obs_shape=shape, features_dim=conv_features_dim)]
        if head_hidden_sizes or conv_features_dim != embedding_dim:
            encoder_layers.append(
                _build_mlp(
                    input_dim=conv_features_dim,
                    hidden_sizes=head_hidden_sizes,
                    output_dim=embedding_dim,
                    activation=activation,
                )
            )
        return nn.Sequential(*encoder_layers)

    raise ValueError(f"RNDModel expects flat or image observations, got {shape!r}")


class RNDModel(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (256,),
        embedding_dim: int = 128,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.embedding_dim = int(embedding_dim)
        self.predictor = _build_rnd_encoder(
            obs_shape=self.obs_shape,
            hidden_sizes=hidden_sizes,
            embedding_dim=self.embedding_dim,
            activation=activation,
        )
        self.target = _build_rnd_encoder(
            obs_shape=self.obs_shape,
            hidden_sizes=hidden_sizes,
            embedding_dim=self.embedding_dim,
            activation=activation,
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def _flatten_obs(self, obs: object, *, device: torch.device | None = None) -> tuple[torch.Tensor, tuple[int, ...]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.ndim < len(self.obs_shape):
            raise ValueError(
                "RNDModel received observation with fewer dimensions than expected: "
                f"obs_shape={tuple(obs_tensor.shape)!r}, expected suffix={self.obs_shape!r}"
            )

        suffix = tuple(int(dim) for dim in obs_tensor.shape[-len(self.obs_shape) :])
        if suffix != self.obs_shape:
            raise ValueError(
                f"RNDModel expected observation suffix {self.obs_shape!r}, got {tuple(obs_tensor.shape)!r}"
            )

        leading_shape = tuple(int(dim) for dim in obs_tensor.shape[: -len(self.obs_shape)])
        if not leading_shape:
            flat_obs = obs_tensor.reshape(1, *self.obs_shape)
        else:
            flat_obs = obs_tensor.reshape(-1, *self.obs_shape)
        return flat_obs, leading_shape

    def prediction_error(self, obs: object) -> torch.Tensor:
        device = next(self.predictor.parameters()).device
        flat_obs, leading_shape = self._flatten_obs(obs, device=device)
        predictor_embedding = self.predictor(flat_obs)
        with torch.no_grad():
            target_embedding = self.target(flat_obs)
        error = (predictor_embedding - target_embedding).pow(2).mean(dim=-1)
        if not leading_shape:
            return error.reshape(())
        return error.reshape(*leading_shape)

    @torch.no_grad()
    def intrinsic_reward(self, obs: object) -> torch.Tensor:
        return self.prediction_error(obs)
