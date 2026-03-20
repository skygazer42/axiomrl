from __future__ import annotations

from rl_training.experiment.registry_core import AlgorithmSpec
from rl_training.experiment.registry_providers import ALL_SPEC_GROUPS


def _merge_spec_groups() -> dict[str, AlgorithmSpec]:
    merged: dict[str, AlgorithmSpec] = {}
    for group_name, group_specs in ALL_SPEC_GROUPS.items():
        duplicate_names = set(merged).intersection(group_specs)
        if duplicate_names:
            duplicate_list = ", ".join(sorted(duplicate_names))
            raise ValueError(f"duplicate algorithm specs in group {group_name!r}: {duplicate_list}")
        merged.update(group_specs)
    return merged


_ALGORITHM_REGISTRY = _merge_spec_groups()


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm: {name!r}") from exc


def list_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(_ALGORITHM_REGISTRY.values())


__all__ = ["AlgorithmSpec", "get_algorithm_spec", "list_algorithm_specs"]
