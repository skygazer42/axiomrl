from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from axiomrl.data.offline_dataset import TransitionDataset


def _sample_dataset_indices(
    rng: np.random.Generator,
    *,
    dataset_size: int,
    count: int,
) -> torch.Tensor:
    return torch.as_tensor(
        rng.integers(0, dataset_size, size=count),
        dtype=torch.int64,
    )


def _append_sampled_transition_parts(
    *,
    dataset: TransitionDataset,
    indices: torch.Tensor,
    obs_parts: list[torch.Tensor],
    action_parts: list[torch.Tensor],
    reward_parts: list[torch.Tensor],
    next_obs_parts: list[torch.Tensor],
    done_parts: list[torch.Tensor],
    next_action_parts: list[torch.Tensor],
    returns_to_go_parts: list[torch.Tensor],
    include_next_actions: bool,
    include_returns_to_go: bool,
) -> None:
    obs_parts.append(dataset.obs.index_select(0, indices))
    action_parts.append(dataset.actions.index_select(0, indices))
    reward_parts.append(dataset.rewards.index_select(0, indices))
    next_obs_parts.append(dataset.next_obs.index_select(0, indices))
    done_parts.append(dataset.dones.index_select(0, indices))

    if include_next_actions:
        source_next_actions = dataset.next_actions if dataset.next_actions is not None else dataset.actions
        next_action_parts.append(source_next_actions.index_select(0, indices))
    if include_returns_to_go:
        assert dataset.returns_to_go is not None
        returns_to_go_parts.append(dataset.returns_to_go.index_select(0, indices))


def _cat_and_permute(parts: list[torch.Tensor], *, permutation: torch.Tensor) -> torch.Tensor:
    return torch.cat(parts, dim=0).index_select(0, permutation)


def _resolve_optional_mixed_field(
    parts: list[torch.Tensor],
    *,
    include_field: bool,
    permutation: torch.Tensor,
) -> torch.Tensor | None:
    if not include_field:
        return None
    return _cat_and_permute(parts, permutation=permutation)


def _validate_dataset_compatibility(datasets: Sequence[TransitionDataset]) -> None:
    reference = datasets[0]
    for dataset in datasets[1:]:
        if tuple(dataset.obs.shape[1:]) != tuple(reference.obs.shape[1:]):
            raise ValueError("all mixed datasets must share the same observation shape")
        if tuple(dataset.actions.shape[1:]) != tuple(reference.actions.shape[1:]):
            raise ValueError("all mixed datasets must share the same action shape")
        if dataset.actions.dtype != reference.actions.dtype:
            raise ValueError("all mixed datasets must share the same action dtype")


def _resolve_mix_weights(
    datasets: Sequence[TransitionDataset],
    *,
    weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if weights is None:
        return tuple(1.0 / len(datasets) for _ in datasets)

    if len(weights) != len(datasets):
        raise ValueError("weights must have the same length as datasets")

    raw_weights = tuple(float(weight) for weight in weights)
    if any(weight < 0.0 for weight in raw_weights):
        raise ValueError("weights must be >= 0")

    weight_sum = float(sum(raw_weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return tuple(weight / weight_sum for weight in raw_weights)


def _resolve_mixed_total_size(
    datasets: Sequence[TransitionDataset],
    *,
    total_size: int | None,
) -> int:
    resolved_total_size = int(total_size) if total_size is not None else sum(len(dataset) for dataset in datasets)
    if resolved_total_size < 1:
        raise ValueError(f"total_size must be >= 1, got {resolved_total_size}")
    return resolved_total_size


def _validate_mix_inputs(
    datasets: Sequence[TransitionDataset],
    *,
    weights: Sequence[float] | None,
    total_size: int | None,
) -> tuple[tuple[float, ...], int]:
    if not datasets:
        raise ValueError("mix_transition_datasets requires at least one dataset")
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("cannot mix empty datasets")

    _validate_dataset_compatibility(datasets)
    normalized_weights = _resolve_mix_weights(datasets, weights=weights)
    resolved_total_size = _resolve_mixed_total_size(datasets, total_size=total_size)
    return normalized_weights, resolved_total_size


def mix_transition_datasets(
    datasets: Sequence[TransitionDataset],
    *,
    weights: Sequence[float] | None = None,
    total_size: int | None = None,
    seed: int | None = None,
) -> TransitionDataset:
    normalized_weights, resolved_total_size = _validate_mix_inputs(
        datasets,
        weights=weights,
        total_size=total_size,
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(resolved_total_size, normalized_weights)

    obs_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    next_obs_parts: list[torch.Tensor] = []
    done_parts: list[torch.Tensor] = []
    next_action_parts: list[torch.Tensor] = []
    returns_to_go_parts: list[torch.Tensor] = []
    include_next_actions = any(dataset.next_actions is not None for dataset in datasets)
    include_returns_to_go = all(dataset.returns_to_go is not None for dataset in datasets)

    for dataset, count in zip(datasets, counts, strict=True):
        if int(count) == 0:
            continue
        indices = _sample_dataset_indices(
            rng,
            dataset_size=len(dataset),
            count=int(count),
        )
        _append_sampled_transition_parts(
            dataset=dataset,
            indices=indices,
            obs_parts=obs_parts,
            action_parts=action_parts,
            reward_parts=reward_parts,
            next_obs_parts=next_obs_parts,
            done_parts=done_parts,
            next_action_parts=next_action_parts,
            returns_to_go_parts=returns_to_go_parts,
            include_next_actions=include_next_actions,
            include_returns_to_go=include_returns_to_go,
        )

    permutation = torch.as_tensor(rng.permutation(resolved_total_size), dtype=torch.int64)
    next_actions = _resolve_optional_mixed_field(
        next_action_parts,
        include_field=include_next_actions,
        permutation=permutation,
    )
    returns_to_go = _resolve_optional_mixed_field(
        returns_to_go_parts,
        include_field=include_returns_to_go,
        permutation=permutation,
    )
    return TransitionDataset(
        obs=_cat_and_permute(obs_parts, permutation=permutation),
        actions=_cat_and_permute(action_parts, permutation=permutation),
        rewards=_cat_and_permute(reward_parts, permutation=permutation),
        next_obs=_cat_and_permute(next_obs_parts, permutation=permutation),
        dones=_cat_and_permute(done_parts, permutation=permutation),
        next_actions=next_actions,
        returns_to_go=returns_to_go,
    )
