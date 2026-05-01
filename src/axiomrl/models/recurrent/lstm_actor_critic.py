from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from axiomrl.models.cnn.nature import NatureCNN
from axiomrl.models.mlp_actor_critic import _build_mlp
from axiomrl.policies.base import PolicyOutput

LSTMState = tuple[torch.Tensor, torch.Tensor]


class LSTMActorCritic(nn.Module):
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
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            raise ValueError(f"LSTMActorCritic expects flat or image observations, got {self.obs_shape!r}")

        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.actor = _build_mlp(
            input_dim=hidden_size,
            hidden_sizes=head_hidden_sizes,
            output_dim=action_dim,
            activation=activation,
        )
        self.critic = _build_mlp(
            input_dim=hidden_size,
            hidden_sizes=head_hidden_sizes,
            output_dim=1,
            activation=activation,
        )

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> LSTMState:
        state_device = device or next(self.parameters()).device
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=state_device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=state_device)
        return hidden, cell

    def reset_state(self, state: LSTMState | None, episode_starts: torch.Tensor | None) -> LSTMState:
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
        if len(self.obs_shape) == 1:
            return self.encoder(obs)
        return self.encoder(obs)

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        obs_tensor = self._prepare_obs(obs)
        batch_size = int(obs_tensor.shape[0])
        recurrent_state = state if isinstance(state, tuple) else None
        if recurrent_state is None:
            recurrent_state = self.initial_state(batch_size, device=obs_tensor.device)

        encoded = self._encode_obs(obs_tensor)
        outputs, next_state = self.lstm(encoded.unsqueeze(1), recurrent_state)
        latent = outputs.squeeze(1)
        distribution = Categorical(logits=self.actor(latent))
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        logprobs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(latent).squeeze(-1)
        return PolicyOutput(
            actions=actions,
            logprobs=logprobs,
            values=values,
            entropy=entropy,
            state=next_state,
        )

    def evaluate_actions_sequence(
        self,
        obs: object,
        actions: object,
        *,
        initial_state: LSTMState,
        episode_starts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim != len(self.obs_shape) + 2:
            raise ValueError(f"expected recurrent obs shape (time, batch, ...), got {tuple(obs_tensor.shape)!r}")

        time_steps, batch_size = int(obs_tensor.shape[0]), int(obs_tensor.shape[1])
        flat_obs = obs_tensor.reshape(time_steps * batch_size, *self.obs_shape)
        encoded = self._encode_obs(flat_obs).reshape(time_steps, batch_size, -1)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=encoded.device)
        episode_start_tensor = torch.as_tensor(episode_starts, dtype=torch.bool, device=encoded.device)

        state = initial_state
        logits_seq: list[torch.Tensor] = []
        values_seq: list[torch.Tensor] = []

        for step in range(time_steps):
            state = self.reset_state(state, episode_start_tensor[step])
            outputs, state = self.lstm(encoded[step].unsqueeze(1), state)
            latent = outputs.squeeze(1)
            logits_seq.append(self.actor(latent))
            values_seq.append(self.critic(latent).squeeze(-1))

        logits = torch.stack(logits_seq, dim=0)
        values = torch.stack(values_seq, dim=0)
        distribution = Categorical(logits=logits)
        return {
            "logprobs": distribution.log_prob(action_tensor),
            "entropy": distribution.entropy(),
            "values": values,
        }
