from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from rl_training.data.offline_dataset import TransitionDataset


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

    reference = datasets[0]
    for dataset in datasets[1:]:
        if tuple(dataset.obs.shape[1:]) != tuple(reference.obs.shape[1:]):
            raise ValueError("all mixed datasets must share the same observation shape")
        if tuple(dataset.actions.shape[1:]) != tuple(reference.actions.shape[1:]):
            raise ValueError("all mixed datasets must share the same action shape")
        if dataset.actions.dtype != reference.actions.dtype:
            raise ValueError("all mixed datasets must share the same action dtype")

    if weights is None:
        normalized_weights = tuple(1.0 / len(datasets) for _ in datasets)
    else:
        if len(weights) != len(datasets):
            raise ValueError("weights must have the same length as datasets")
        raw_weights = tuple(float(weight) for weight in weights)
        if any(weight < 0.0 for weight in raw_weights):
            raise ValueError("weights must be >= 0")
        weight_sum = float(sum(raw_weights))
        if weight_sum <= 0.0:
            raise ValueError("weights must sum to a positive value")
        normalized_weights = tuple(weight / weight_sum for weight in raw_weights)

    resolved_total_size = int(total_size) if total_size is not None else sum(len(dataset) for dataset in datasets)
    if resolved_total_size < 1:
        raise ValueError(f"total_size must be >= 1, got {resolved_total_size}")
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
        indices = torch.as_tensor(
            rng.integers(0, len(dataset), size=int(count)),
            dtype=torch.int64,
        )
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

    mixed_obs = torch.cat(obs_parts, dim=0)
    mixed_actions = torch.cat(action_parts, dim=0)
    mixed_rewards = torch.cat(reward_parts, dim=0)
    mixed_next_obs = torch.cat(next_obs_parts, dim=0)
    mixed_dones = torch.cat(done_parts, dim=0)

    permutation = torch.as_tensor(rng.permutation(resolved_total_size), dtype=torch.int64)
    next_actions = torch.cat(next_action_parts, dim=0).index_select(0, permutation) if include_next_actions else None
    returns_to_go = (
        torch.cat(returns_to_go_parts, dim=0).index_select(0, permutation) if include_returns_to_go else None
    )
    return TransitionDataset(
        obs=mixed_obs.index_select(0, permutation),
        actions=mixed_actions.index_select(0, permutation),
        rewards=mixed_rewards.index_select(0, permutation),
        next_obs=mixed_next_obs.index_select(0, permutation),
        dones=mixed_dones.index_select(0, permutation),
        next_actions=next_actions,
        returns_to_go=returns_to_go,
    )
