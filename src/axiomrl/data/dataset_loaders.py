from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from axiomrl.data.offline_dataset import TransitionDataset

_UNTRUSTED_TORCH_DATASET_ERROR = (
    "dataset_kind='pt'/'pth'/'torch' is unsafe for untrusted datasets; "
    "convert the dataset to .npz or use Minari instead"
)


def _resolve_dataset_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("dataset_path must be provided for file-backed datasets")
    return Path(path)


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _iterate_minari_episodes(dataset: object) -> Iterable[object]:
    if hasattr(dataset, "iterate_episodes"):
        return dataset.iterate_episodes()  # type: ignore[no-any-return]
    if hasattr(dataset, "episodes"):
        return dataset.episodes  # type: ignore[no-any-return]
    raise TypeError("unsupported Minari dataset object: missing episode iterator")


def _load_minari_payload(*, dataset_id: str, download: bool) -> dict[str, Any]:
    try:
        import minari
    except ImportError as exc:
        raise ImportError("dataset_kind='minari' requires the optional 'minari' dependency") from exc

    dataset = minari.load_dataset(dataset_id, download=download)

    obs_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    reward_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    done_parts: list[np.ndarray] = []
    next_action_parts: list[np.ndarray] = []

    for episode in _iterate_minari_episodes(dataset):
        observations = np.asarray(getattr(episode, "observations"), dtype=np.float32)
        actions = np.asarray(getattr(episode, "actions"))
        rewards = np.asarray(getattr(episode, "rewards"), dtype=np.float32)
        terminations = np.asarray(getattr(episode, "terminations"), dtype=np.float32)
        truncations = np.asarray(getattr(episode, "truncations"), dtype=np.float32)
        next_actions = np.concatenate([actions[1:], actions[-1:]], axis=0)

        obs_parts.append(observations[:-1])
        action_parts.append(actions)
        reward_parts.append(rewards)
        next_obs_parts.append(observations[1:])
        done_parts.append(np.logical_or(terminations, truncations).astype(np.float32))
        next_action_parts.append(next_actions)

    if not obs_parts:
        raise ValueError(f"Minari dataset {dataset_id!r} does not contain any episodes")

    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
        "rewards": np.concatenate(reward_parts, axis=0),
        "next_obs": np.concatenate(next_obs_parts, axis=0),
        "dones": np.concatenate(done_parts, axis=0),
        "next_actions": np.concatenate(next_action_parts, axis=0),
    }


def load_transition_dataset(
    kind: str,
    *,
    dataset_path: str | Path | None = None,
    dataset_id: str | None = None,
    download: bool = False,
) -> TransitionDataset:
    normalized_kind = str(kind).lower()

    if normalized_kind == "npz":
        payload = _load_npz_payload(_resolve_dataset_path(dataset_path))
    elif normalized_kind in {"pt", "pth", "torch"}:
        _resolve_dataset_path(dataset_path)
        raise ValueError(_UNTRUSTED_TORCH_DATASET_ERROR)
    elif normalized_kind == "minari":
        if not dataset_id:
            raise ValueError("dataset_id must be provided for dataset_kind='minari'")
        payload = _load_minari_payload(dataset_id=dataset_id, download=download)
    else:
        raise ValueError(f"unsupported dataset kind: {kind!r}")

    return TransitionDataset.from_dict(payload)
