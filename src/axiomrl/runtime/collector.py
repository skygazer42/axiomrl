from dataclasses import dataclass
from typing import Protocol

from axiomrl.runtime.types import MetricDict


@dataclass(slots=True)
class CollectResult:
    num_env_steps: int
    num_episodes: int
    metrics: MetricDict
    last_obs: object | None = None


class Collector(Protocol):
    def reset(self) -> None: ...

    def collect_steps(
        self,
        *,
        num_steps: int,
        deterministic: bool = False,
    ) -> CollectResult: ...

    def collect_episodes(
        self,
        *,
        num_episodes: int,
        deterministic: bool = False,
    ) -> CollectResult: ...
