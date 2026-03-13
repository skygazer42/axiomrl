from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from rl_training.models.cnn.nature import NatureCNN
from rl_training.models.mlp_q_network import _build_mlp


LSTMState = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class RecurrentQOutput:
    actions: torch.Tensor
    q_values: torch.Tensor
    state: LSTMState


class LSTMQNetwork(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: Sequence[int],
        action_dim: int,
        features_dim: int = 256,
        encoder_hidden_sizes: Sequence[int] = (128,),
        head_hidden_sizes: Sequence[int] = (128,),
        hidden_size: int = 256,
        num_layers: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.action_dim = int(action_dim)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        if len(self.obs_shape) == 1:
            self.encoder = _build_mlp(
                input_dim=self.obs_shape[0],
                hidden_sizes=encoder_hidden_sizes,
                output_dim=features_dim,
                activation=activation,
            )
        elif len(self.obs_shape) == 3:
            self.encoder = NatureCNN(obs_shape=self.obs_shape, features_dim=features_dim)
        else:
            raise ValueError(f"LSTMQNetwork expects flat or image observations, got {self.obs_shape!r}")

        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.q_head = _build_mlp(
            input_dim=self.hidden_size,
            hidden_sizes=head_hidden_sizes,
            output_dim=self.action_dim,
            activation=activation,
        )

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> LSTMState:
        state_device = device or next(self.parameters()).device
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=state_device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=state_device)
        return hidden, cell

    def reset_state(self, state: LSTMState | None, episode_starts: object | None) -> LSTMState:
        if episode_starts is None:
            if state is None:
                raise ValueError("episode_starts and state cannot both be None")
            return state

        episode_start_tensor = torch.as_tensor(episode_starts, dtype=torch.bool)
        if episode_start_tensor.ndim == 0:
            episode_start_tensor = episode_start_tensor.unsqueeze(0)

        if state is None:
            return self.initial_state(int(episode_start_tensor.shape[0]), device=episode_start_tensor.device)

        hidden, cell = state
        mask = (~episode_start_tensor).to(dtype=hidden.dtype, device=hidden.device).view(1, -1, 1)
        return hidden * mask, cell * mask

    def _prepare_obs(self, obs: object) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == len(self.obs_shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _forward_step(
        self,
        obs: object,
        *,
        state: LSTMState | None = None,
        episode_starts: object | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        obs_tensor = self._prepare_obs(obs)
        batch_size = int(obs_tensor.shape[0])
        recurrent_state = state
        if recurrent_state is None:
            recurrent_state = self.initial_state(batch_size, device=obs_tensor.device)
        recurrent_state = self.reset_state(recurrent_state, episode_starts)
        encoded = self._encode_obs(obs_tensor)
        outputs, next_state = self.lstm(encoded.unsqueeze(1), recurrent_state)
        return self.q_head(outputs.squeeze(1)), next_state

    def act(
        self,
        obs: object,
        *,
        state: LSTMState | None = None,
        epsilon: float = 0.0,
        deterministic: bool = False,
        episode_starts: object | None = None,
    ) -> RecurrentQOutput:
        q_values, next_state = self._forward_step(obs, state=state, episode_starts=episode_starts)
        greedy_actions = q_values.argmax(dim=-1)

        if deterministic or epsilon <= 0.0:
            actions = greedy_actions
        else:
            random_actions = torch.randint(
                0,
                self.action_dim,
                greedy_actions.shape,
                device=greedy_actions.device,
            )
            explore_mask = torch.rand(greedy_actions.shape, device=greedy_actions.device) < float(epsilon)
            actions = torch.where(explore_mask, random_actions, greedy_actions)

        return RecurrentQOutput(actions=actions, q_values=q_values, state=next_state)

    def q_values_sequence(
        self,
        obs: object,
        *,
        initial_state: LSTMState,
        episode_starts: object,
    ) -> torch.Tensor:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim != len(self.obs_shape) + 2:
            raise ValueError(f"expected recurrent obs shape (time, batch, ...), got {tuple(obs_tensor.shape)!r}")

        time_steps, batch_size = int(obs_tensor.shape[0]), int(obs_tensor.shape[1])
        flat_obs = obs_tensor.reshape(time_steps * batch_size, *self.obs_shape)
        encoded = self._encode_obs(flat_obs).reshape(time_steps, batch_size, -1)
        episode_start_tensor = torch.as_tensor(episode_starts, dtype=torch.bool, device=encoded.device)

        state = initial_state
        q_values_sequence: list[torch.Tensor] = []
        for step in range(time_steps):
            state = self.reset_state(state, episode_start_tensor[step])
            outputs, state = self.lstm(encoded[step].unsqueeze(1), state)
            q_values_sequence.append(self.q_head(outputs.squeeze(1)))

        return torch.stack(q_values_sequence, dim=0)
