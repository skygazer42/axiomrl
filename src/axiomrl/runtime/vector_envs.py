from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym

EnvFactory = Callable[[], gym.Env]

_SUPPORTED_EXECUTION_BACKENDS = ("local_sync", "local_async")


class WorkerBackend(Protocol):
    name: str

    def make_vector_env(self, env_fns: Sequence[EnvFactory]) -> gym.vector.VectorEnv: ...


@dataclass(frozen=True, slots=True)
class LocalSyncBackend:
    name: str = "local_sync"

    def make_vector_env(self, env_fns: Sequence[EnvFactory]) -> gym.vector.SyncVectorEnv:
        return gym.vector.SyncVectorEnv(list(env_fns))


@dataclass(frozen=True, slots=True)
class LocalAsyncBackend:
    name: str = "local_async"

    def make_vector_env(self, env_fns: Sequence[EnvFactory]) -> gym.vector.AsyncVectorEnv:
        return gym.vector.AsyncVectorEnv(list(env_fns))


def supported_execution_backends() -> tuple[str, ...]:
    return _SUPPORTED_EXECUTION_BACKENDS


def resolve_worker_backend(name: str) -> WorkerBackend:
    if name == "local_sync":
        return LocalSyncBackend()
    if name == "local_async":
        return LocalAsyncBackend()
    supported = ", ".join(_SUPPORTED_EXECUTION_BACKENDS)
    raise ValueError(f"unsupported execution backend {name!r}; expected one of: {supported}")
