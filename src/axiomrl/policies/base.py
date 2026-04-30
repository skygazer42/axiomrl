from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class PolicyOutput:
    actions: object | None
    logprobs: object | None
    values: object | None
    entropy: object | None
    state: object | None


class Policy(Protocol):
    def train(self, mode: bool = True) -> Policy: ...

    def eval(self) -> Policy: ...

    def act(
        self,
        obs: object,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput: ...

    def parameters(self) -> Iterator[object]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...
