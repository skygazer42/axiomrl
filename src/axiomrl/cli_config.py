from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from rl_training.experiment.config import TrainConfig
from rl_training.resources import find_packaged_asset


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read YAML config at {path}: {exc}") from exc

    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def _resolve_linked_config_path(config_path: Path, linked_path: str) -> Path:
    candidate = Path(linked_path)
    if candidate.is_absolute():
        return candidate

    for base_dir in config_path.parents:
        parent_candidate = (base_dir / candidate).resolve()
        if parent_candidate.exists():
            return parent_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (config_path.parent / candidate).resolve()


def _load_config_payload(path: Path, *, visited: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    seen = set() if visited is None else set(visited)
    if resolved_path in seen:
        raise ValueError(f"detected config include cycle at {resolved_path}")

    seen.add(resolved_path)
    payload = _load_yaml_mapping(resolved_path)
    if "algo" in payload:
        return payload

    linked_config = payload.get("config")
    if isinstance(linked_config, str):
        from rl_training.zoo.manifests import apply_manifest_defaults_to_config_payload

        linked_path = _resolve_linked_config_path(resolved_path, linked_config)
        linked_payload = _load_config_payload(linked_path, visited=seen)
        return apply_manifest_defaults_to_config_payload(
            linked_payload, preset_path=resolved_path, preset_payload=payload
        )

    raise ValueError(f"config file {resolved_path} must define 'algo' or reference another config via 'config'")


def _resolve_input_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    packaged = find_packaged_asset(candidate)
    if packaged is not None:
        return packaged
    return candidate


def _require_payload_field(payload: Mapping[str, Any], field: str, *, config_path: Path) -> object:
    if field not in payload:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    return payload[field]


def _coerce_required_str(value: object, field: str, *, config_path: Path) -> str:
    if value is None:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return normalized


def _coerce_optional_str(payload: Mapping[str, Any], field: str, default: str, *, config_path: Path) -> str:
    value = payload.get(field)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return normalized


def _coerce_int(value: object, field: str, *, config_path: Path) -> int:
    if isinstance(value, bool):
        raise TypeError(f"config file {config_path} field '{field}' must be an integer, got {type(value)!r}")
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"config file {config_path} field '{field}' must be an integer, got {type(value)!r}") from exc


def _coerce_required_int(
    payload: Mapping[str, Any], field: str, *, config_path: Path, min_value: int | None = None
) -> int:
    value = _require_payload_field(payload, field, config_path=config_path)
    int_value = _coerce_int(value, field, config_path=config_path)
    if min_value is not None and int_value < min_value:
        raise ValueError(f"config file {config_path} field '{field}' must be >= {min_value}, got {int_value}")
    return int_value


def _coerce_optional_int(
    payload: Mapping[str, Any],
    field: str,
    default: int,
    *,
    config_path: Path,
    min_value: int | None = None,
) -> int:
    value = payload.get(field)
    if value is None:
        return default
    int_value = _coerce_int(value, field, config_path=config_path)
    if min_value is not None and int_value < min_value:
        raise ValueError(f"config file {config_path} field '{field}' must be >= {min_value}, got {int_value}")
    return int_value


def _coerce_required_path(value: object, field: str, *, config_path: Path) -> Path:
    if value is None:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string path, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return Path(normalized)


def _coerce_optional_mapping(
    payload: Mapping[str, Any],
    field: str,
    *,
    config_path: Path,
) -> dict[str, Any]:
    value = payload.get(field)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config file {config_path} field '{field}' must be a mapping, got {type(value)!r}")
    return dict(value)


def _coerce_optional_tags(payload: Mapping[str, Any], *, config_path: Path) -> tuple[str, ...]:
    value = payload.get("tags")
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise TypeError(f"config file {config_path} field 'tags' must be a sequence of strings, got {type(value)!r}")
    if not isinstance(value, Sequence):
        raise TypeError(f"config file {config_path} field 'tags' must be a sequence of strings, got {type(value)!r}")
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"config file {config_path} field 'tags' entries must be strings, got {type(entry)!r}")
        tag = entry.strip()
        if tag:
            normalized.append(tag)
    return tuple(normalized)


def load_config(path: str | Path) -> TrainConfig:
    config_path = _resolve_input_config_path(path)
    if not config_path.exists():
        raise ValueError(f"config file {config_path} does not exist")
    payload = _load_config_payload(config_path)

    return TrainConfig(
        algo=_coerce_required_str(
            _require_payload_field(payload, "algo", config_path=config_path), "algo", config_path=config_path
        ),
        env_id=_coerce_required_str(
            _require_payload_field(payload, "env_id", config_path=config_path),
            "env_id",
            config_path=config_path,
        ),
        seed=_coerce_required_int(payload, "seed", config_path=config_path, min_value=0),
        total_timesteps=_coerce_required_int(payload, "total_timesteps", config_path=config_path, min_value=1),
        output_dir=_coerce_required_path(
            _require_payload_field(payload, "output_dir", config_path=config_path),
            "output_dir",
            config_path=config_path,
        ),
        execution_backend=_coerce_optional_str(payload, "execution_backend", "local_sync", config_path=config_path),
        device=_coerce_optional_str(payload, "device", "auto", config_path=config_path),
        num_envs=_coerce_optional_int(payload, "num_envs", 1, config_path=config_path, min_value=1),
        eval_episodes=_coerce_optional_int(payload, "eval_episodes", 5, config_path=config_path, min_value=1),
        log_interval=_coerce_optional_int(payload, "log_interval", 1, config_path=config_path, min_value=1),
        checkpoint_interval=_coerce_optional_int(
            payload, "checkpoint_interval", 1, config_path=config_path, min_value=1
        ),
        tags=_coerce_optional_tags(payload, config_path=config_path),
        benchmark=_coerce_optional_mapping(payload, "benchmark", config_path=config_path),
        algo_kwargs=_coerce_optional_mapping(payload, "algo_kwargs", config_path=config_path),
        env_kwargs=_coerce_optional_mapping(payload, "env_kwargs", config_path=config_path),
    )


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, Any] = {}

    if getattr(args, "output_dir", None) is not None:
        overrides["output_dir"] = Path(args.output_dir)
    if getattr(args, "execution_backend", None) is not None:
        overrides["execution_backend"] = str(args.execution_backend)
    if getattr(args, "total_timesteps", None) is not None:
        overrides["total_timesteps"] = int(args.total_timesteps)
    if getattr(args, "num_envs", None) is not None:
        overrides["num_envs"] = int(args.num_envs)
    if getattr(args, "eval_episodes", None) is not None:
        overrides["eval_episodes"] = int(args.eval_episodes)
    if getattr(args, "seeds", None) is not None:
        raw_value = str(args.seeds)
        tokens = [token.strip() for token in raw_value.split(",")]
        if any(token == "" for token in tokens):
            raise ValueError("--seeds expects a comma-separated list of integers, for example: 1,2,3")
        try:
            seed_values = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("--seeds expects a comma-separated list of integers, for example: 1,2,3") from exc
        if any(seed < 0 for seed in seed_values):
            raise ValueError("--seeds expects a comma-separated list of non-negative integers, for example: 1,2,3")
        benchmark = dict(config.benchmark)
        benchmark["seeds"] = seed_values
        overrides["benchmark"] = benchmark

    return cast(TrainConfig, replace(config, **overrides))


def serialize_train_config(config: TrainConfig) -> dict[str, Any]:
    return {
        "algo": config.algo,
        "env_id": config.env_id,
        "seed": config.seed,
        "total_timesteps": config.total_timesteps,
        "output_dir": str(config.output_dir),
        "execution_backend": config.execution_backend,
        "device": config.device,
        "num_envs": config.num_envs,
        "eval_episodes": config.eval_episodes,
        "log_interval": config.log_interval,
        "checkpoint_interval": config.checkpoint_interval,
        "tags": list(config.tags),
        "benchmark": dict(config.benchmark),
        "algo_kwargs": dict(config.algo_kwargs),
        "env_kwargs": dict(config.env_kwargs),
    }
