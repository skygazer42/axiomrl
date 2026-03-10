from __future__ import annotations

from collections import deque
from dataclasses import dataclass


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

