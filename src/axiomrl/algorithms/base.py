from dataclasses import dataclass
from typing import Protocol

from axiomrl.policies.base import Policy
from axiomrl.runtime.types import MetricDict


@dataclass(slots=True)
class UpdateResult:
    metrics: MetricDict
    num_gradient_steps: int


class Algorithm(Protocol):
    policy: Policy

    def update(self, batch: object, *, global_step: int) -> UpdateResult: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> None: ...

    def set_train_mode(self) -> None: ...

    def set_eval_mode(self) -> None: ...
