from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Step:
    obs: object
    action: object
    reward: float
    next_obs: object
    done: bool


class NStepAccumulator:
    def __init__(
        self,
        *,
        num_envs: int,
        n_step: int,
        gamma: float,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        if n_step <= 0:
            raise ValueError(f"n_step must be > 0, got {n_step}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        self.num_envs = int(num_envs)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._buffers: list[deque[_Step]] = [deque() for _ in range(self.num_envs)]

    def add(
        self,
        env_index: int,
        obs: object,
        action: object,
        reward: object,
        next_obs: object,
        done: object,
    ) -> list[dict[str, object]]:
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index out of range: {env_index}")

        buffer = self._buffers[env_index]
        step = _Step(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
        )
        buffer.append(step)

        transitions: list[dict[str, object]] = []

        if step.done:
            while buffer:
                transitions.append(self._pop_transition(buffer))
            return transitions

        while len(buffer) >= self.n_step:
            transitions.append(self._pop_transition(buffer))

        return transitions

    def _pop_transition(self, buffer: deque[_Step]) -> dict[str, object]:
        first = buffer[0]
        reward_sum = 0.0
        last_next_obs = first.next_obs
        last_done = False

        for step_index, step in enumerate(buffer):
            reward_sum += (self.gamma**step_index) * step.reward
            last_next_obs = step.next_obs
            last_done = step.done
            if step.done or (step_index + 1) >= self.n_step:
                break

        buffer.popleft()
        return {
            "obs": first.obs,
            "actions": first.action,
            "rewards": reward_sum,
            "next_obs": last_next_obs,
            "dones": float(last_done),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "num_envs": self.num_envs,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "buffers": [
                [
                    {
                        "obs": _serialize_step_value(step.obs),
                        "action": _serialize_step_value(step.action),
                        "reward": float(step.reward),
                        "next_obs": _serialize_step_value(step.next_obs),
                        "done": bool(step.done),
                    }
                    for step in buffer
                ]
                for buffer in self._buffers
            ],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        serialized_buffers = state_dict.get("buffers", [])
        if not isinstance(serialized_buffers, list) or len(serialized_buffers) != self.num_envs:
            raise ValueError("n-step accumulator buffer count does not match num_envs")

        self._buffers = [deque() for _ in range(self.num_envs)]
        for env_index, serialized_buffer in enumerate(serialized_buffers):
            if not isinstance(serialized_buffer, list):
                raise TypeError(f"expected serialized n-step buffer list, got {type(serialized_buffer)!r}")
            buffer = self._buffers[env_index]
            for step_payload in serialized_buffer:
                if not isinstance(step_payload, dict):
                    raise TypeError(f"expected serialized n-step step dict, got {type(step_payload)!r}")
                buffer.append(
                    _Step(
                        obs=_deserialize_step_value(step_payload["obs"]),
                        action=_deserialize_step_value(step_payload["action"]),
                        reward=float(step_payload["reward"]),
                        next_obs=_deserialize_step_value(step_payload["next_obs"]),
                        done=bool(step_payload["done"]),
                    )
                )


def _serialize_step_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {
            "__n_step_kind__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__n_step_kind__": "tuple", "items": [_serialize_step_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_step_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_step_value(item) for key, item in value.items()}
    return value


def _deserialize_step_value(value: object) -> object:
    if isinstance(value, dict):
        kind = value.get("__n_step_kind__")
        if kind == "ndarray":
            return np.asarray(value["data"], dtype=np.dtype(str(value["dtype"])))
        if kind == "tuple":
            return tuple(_deserialize_step_value(item) for item in value["items"])
        return {key: _deserialize_step_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_step_value(item) for item in value]
    return value
