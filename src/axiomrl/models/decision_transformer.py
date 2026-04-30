from __future__ import annotations

import torch
from torch import nn


class DecisionTransformerModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        context_length: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_timestep: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if max_timestep < 1:
            raise ValueError("max_timestep must be positive")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.context_length = int(context_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.max_timestep = int(max_timestep)

        self.obs_embedding = nn.Linear(self.obs_dim, self.hidden_size)
        self.action_embedding = nn.Linear(self.action_dim, self.hidden_size)
        self.return_embedding = nn.Linear(1, self.hidden_size)
        self.timestep_embedding = nn.Embedding(self.max_timestep + 1, self.hidden_size)
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.input_dropout = nn.Dropout(float(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim),
            nn.Tanh(),
        )

    def _prepare_model_inputs(
        self,
        *,
        obs: object,
        actions: object,
        returns_to_go: object,
        timesteps: object,
        mask: object | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        returns_tensor = torch.as_tensor(returns_to_go, dtype=torch.float32, device=device)
        timestep_tensor = torch.as_tensor(timesteps, dtype=torch.int64, device=device)
        if mask is None:
            mask_tensor = None
        else:
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)

        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        if returns_tensor.ndim == 1:
            returns_tensor = returns_tensor.unsqueeze(0)
        if timestep_tensor.ndim == 1:
            timestep_tensor = timestep_tensor.unsqueeze(0)
        if mask_tensor is not None and mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        if mask_tensor is None:
            mask_tensor = torch.ones(obs_tensor.shape[:2], dtype=torch.float32, device=device)

        return obs_tensor, action_tensor, returns_tensor, timestep_tensor, mask_tensor

    def predict_actions(
        self,
        *,
        obs: object,
        actions: object,
        returns_to_go: object,
        timesteps: object,
        mask: object | None = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        obs_tensor, action_tensor, returns_tensor, timestep_tensor, mask_tensor = self._prepare_model_inputs(
            obs=obs,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            mask=mask,
            device=device,
        )

        if returns_tensor.shape[-1] != 1:
            returns_tensor = returns_tensor.unsqueeze(-1)

        prev_actions = torch.zeros_like(action_tensor)
        prev_actions[:, 1:, :] = action_tensor[:, :-1, :]

        timestep_ids = timestep_tensor.clamp(min=0, max=self.max_timestep)
        tokens = (
            self.obs_embedding(obs_tensor)
            + self.action_embedding(prev_actions)
            + self.return_embedding(returns_tensor)
            + self.timestep_embedding(timestep_ids)
        )
        tokens = self.input_norm(tokens)
        tokens = self.input_dropout(tokens)
        tokens = tokens * mask_tensor.unsqueeze(-1)

        sequence_length = int(tokens.shape[1])
        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device),
            diagonal=1,
        )
        padding_mask = mask_tensor <= 0.0
        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        encoded = self.output_norm(encoded)
        return self.action_head(encoded)

    def predict_last_action(
        self,
        *,
        obs: object,
        actions: object,
        returns_to_go: object,
        timesteps: object,
        mask: object | None = None,
    ) -> torch.Tensor:
        predictions = self.predict_actions(
            obs=obs,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            mask=mask,
        )
        if mask is None:
            return predictions[:, -1, :]

        device = predictions.device
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        valid_tokens = mask_tensor > 0.0
        any_valid = valid_tokens.any(dim=1)
        reversed_offsets = torch.argmax(valid_tokens.to(dtype=torch.int64).flip(dims=(1,)), dim=1)
        default_index = torch.full((predictions.shape[0],), predictions.shape[1] - 1, dtype=torch.int64, device=device)
        last_indices = torch.where(
            any_valid,
            torch.full_like(default_index, predictions.shape[1] - 1) - reversed_offsets,
            default_index,
        )
        batch_indices = torch.arange(predictions.shape[0], device=device)
        return predictions[batch_indices, last_indices]
