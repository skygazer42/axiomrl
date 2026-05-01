from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from axiomrl.cli_config import load_config
from axiomrl.experiment.config import TrainConfig
from axiomrl.resources import find_packaged_asset

_ALLOWED_TOP_LEVEL_PATHS = {
    "total_timesteps",
    "num_envs",
    "eval_episodes",
    "log_interval",
    "checkpoint_interval",
    "device",
    "execution_backend",
}
_FORBIDDEN_PATHS = {"algo", "env_id", "seed", "output_dir", "tags", "benchmark.seeds"}


@dataclass(frozen=True, slots=True)
class StudyObjective:
    metric: str
    mode: str


@dataclass(frozen=True, slots=True)
class StudyOptions:
    name: str
    backend: str
    sampler: str
    num_trials: int | None
    seed: int
    fail_fast: bool
    objective: StudyObjective


@dataclass(frozen=True, slots=True)
class SearchSpaceSpec:
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    values: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class StudyConfig:
    config_path: Path
    base_config_path: Path
    base_train_config: TrainConfig
    output_dir: Path
    study: StudyOptions
    search_space: dict[str, SearchSpaceSpec]


def serialize_study_config(config: StudyConfig) -> dict[str, Any]:
    serialized_search_space: dict[str, dict[str, Any]] = {}
    for path, spec in config.search_space.items():
        payload: dict[str, Any] = {"type": spec.kind}
        if spec.low is not None:
            payload["low"] = spec.low
        if spec.high is not None:
            payload["high"] = spec.high
        if spec.step is not None:
            payload["step"] = spec.step
        if spec.log:
            payload["log"] = True
        if spec.values:
            payload["values"] = list(spec.values)
        serialized_search_space[path] = payload
    return {
        "base_config": str(config.base_config_path),
        "output_dir": str(config.output_dir),
        "study": {
            "name": config.study.name,
            "backend": config.study.backend,
            "sampler": config.study.sampler,
            "num_trials": config.study.num_trials,
            "seed": config.study.seed,
            "fail_fast": config.study.fail_fast,
            "objective": {
                "metric": config.study.objective.metric,
                "mode": config.study.objective.mode,
            },
        },
        "search_space": serialized_search_space,
    }


def _study_config_from_mapping(payload: Mapping[str, Any], *, source_path: Path) -> StudyConfig:
    base_config_raw = payload.get("base_config")
    if base_config_raw is None:
        raise ValueError(f"study config {source_path} missing required key 'base_config'")
    base_config_path = _resolve_relative_path(
        source_path, _require_str(base_config_raw, "base_config", config_path=source_path)
    )
    base_train_config = load_config(base_config_path)

    output_dir_raw = payload.get("output_dir")
    if output_dir_raw is None:
        output_dir = (base_train_config.output_dir / "studies").resolve()
    else:
        output_dir = _resolve_relative_path(
            source_path, _require_str(output_dir_raw, "output_dir", config_path=source_path)
        )

    study_payload = _require_mapping(payload.get("study"), "study", config_path=source_path)
    objective_payload = _require_mapping(study_payload.get("objective"), "study.objective", config_path=source_path)
    objective = StudyObjective(
        metric=_require_str(objective_payload.get("metric"), "study.objective.metric", config_path=source_path),
        mode=_require_str(objective_payload.get("mode"), "study.objective.mode", config_path=source_path),
    )
    if objective.mode not in {"min", "max"}:
        raise ValueError(f"study config {source_path} field 'study.objective.mode' must be 'min' or 'max'")

    backend = _require_str(study_payload.get("backend", "native"), "study.backend", config_path=source_path)
    sampler = _require_str(study_payload.get("sampler", "random"), "study.sampler", config_path=source_path)
    if backend not in {"native", "optuna"}:
        raise ValueError(f"study config {source_path} field 'study.backend' must be 'native' or 'optuna'")
    if backend == "native" and sampler not in {"random", "grid"}:
        raise ValueError(f"study config {source_path} field 'study.sampler' must be 'random' or 'grid'")
    if backend == "optuna" and sampler not in {"random", "tpe"}:
        raise ValueError(f"study config {source_path} field 'study.sampler' must be 'random' or 'tpe'")

    num_trials_raw = study_payload.get("num_trials")
    num_trials = (
        None
        if num_trials_raw is None
        else _require_int(num_trials_raw, "study.num_trials", config_path=source_path, minimum=1)
    )
    if backend == "native" and sampler == "grid" and num_trials is not None:
        raise ValueError(f"study config {source_path} field 'study.num_trials' is not allowed for native grid search")
    if (backend == "native" and sampler == "random") or backend == "optuna":
        if num_trials is None:
            raise ValueError(f"study config {source_path} field 'study.num_trials' is required for this backend")

    seed_raw = study_payload.get("seed", 0)
    seed = _require_int(seed_raw, "study.seed", config_path=source_path, minimum=0)
    fail_fast_raw = study_payload.get("fail_fast", False)
    fail_fast = _require_bool(fail_fast_raw, "study.fail_fast", config_path=source_path)
    search_space = _parse_search_space(
        _require_mapping(payload.get("search_space"), "search_space", config_path=source_path),
        config_path=source_path,
    )

    return StudyConfig(
        config_path=source_path,
        base_config_path=base_config_path,
        base_train_config=base_train_config,
        output_dir=output_dir,
        study=StudyOptions(
            name=_require_str(study_payload.get("name"), "study.name", config_path=source_path),
            backend=backend,
            sampler=sampler,
            num_trials=num_trials,
            seed=seed,
            fail_fast=fail_fast,
            objective=objective,
        ),
        search_space=search_space,
    )


def deserialize_study_config(payload: Mapping[str, Any]) -> StudyConfig:
    return _study_config_from_mapping(payload, source_path=Path("<serialized-study-config>"))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"unable to read study config at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def _resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()

    packaged = find_packaged_asset(candidate)
    if packaged is not None:
        return packaged.resolve()
    return candidate.resolve()


def _resolve_relative_path(config_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()

    local_candidate = (config_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    packaged = find_packaged_asset(candidate)
    if packaged is not None:
        return packaged.resolve()
    return local_candidate


def _require_mapping(value: object, name: str, *, config_path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"study config {config_path} field '{name}' must be a mapping, got {type(value)!r}")
    return value


def _require_str(value: object, name: str, *, config_path: Path) -> str:
    if not isinstance(value, str):
        raise TypeError(f"study config {config_path} field '{name}' must be a string, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"study config {config_path} field '{name}' must not be empty")
    return normalized


def _require_bool(value: object, name: str, *, config_path: Path) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"study config {config_path} field '{name}' must be a boolean, got {type(value)!r}")
    return value


def _require_int(value: object, name: str, *, config_path: Path, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"study config {config_path} field '{name}' must be an integer, got {type(value)!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"study config {config_path} field '{name}' must be >= {minimum}, got {value}")
    return value


def _require_number(value: object, name: str, *, config_path: Path) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"study config {config_path} field '{name}' must be numeric, got {type(value)!r}")
    return value


def _validate_search_path(path: str, *, config_path: Path) -> None:
    if path in _FORBIDDEN_PATHS:
        raise ValueError(f"study config {config_path} search_space path '{path}' is not tunable")
    if path in _ALLOWED_TOP_LEVEL_PATHS:
        return
    if path.startswith("algo_kwargs.") or path.startswith("env_kwargs."):
        return
    if path.startswith("benchmark."):
        if path == "benchmark.seeds":
            raise ValueError(f"study config {config_path} search_space path '{path}' is not tunable")
        return
    raise ValueError(f"study config {config_path} search_space path '{path}' is not supported")


def _parse_categorical_spec(payload: Mapping[str, Any], *, config_path: Path, path: str) -> SearchSpaceSpec:
    values = payload.get("values")
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(
            f"study config {config_path} search_space '{path}' categorical values must be a non-empty sequence"
        )
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"study config {config_path} search_space '{path}' categorical values must not be empty")
    return SearchSpaceSpec(kind="categorical", values=normalized)


def _parse_int_spec(payload: Mapping[str, Any], *, config_path: Path, path: str) -> SearchSpaceSpec:
    low = _require_int(payload.get("low"), f"search_space.{path}.low", config_path=config_path)
    high = _require_int(payload.get("high"), f"search_space.{path}.high", config_path=config_path)
    if low > high:
        raise ValueError(f"study config {config_path} search_space '{path}' low must be <= high")
    step_raw = payload.get("step", 1)
    step = _require_int(step_raw, f"search_space.{path}.step", config_path=config_path, minimum=1)
    log = bool(payload.get("log", False))
    if log and (low <= 0 or high <= 0):
        raise ValueError(f"study config {config_path} search_space '{path}' log ranges must be strictly positive")
    return SearchSpaceSpec(kind="int", low=low, high=high, step=step, log=log)


def _parse_float_spec(payload: Mapping[str, Any], *, config_path: Path, path: str) -> SearchSpaceSpec:
    low = float(_require_number(payload.get("low"), f"search_space.{path}.low", config_path=config_path))
    high = float(_require_number(payload.get("high"), f"search_space.{path}.high", config_path=config_path))
    if low > high:
        raise ValueError(f"study config {config_path} search_space '{path}' low must be <= high")
    step_raw = payload.get("step")
    step = (
        None
        if step_raw is None
        else float(_require_number(step_raw, f"search_space.{path}.step", config_path=config_path))
    )
    if step is not None and step <= 0:
        raise ValueError(f"study config {config_path} search_space '{path}' step must be > 0")
    log = bool(payload.get("log", False))
    if log and (low <= 0 or high <= 0):
        raise ValueError(f"study config {config_path} search_space '{path}' log ranges must be strictly positive")
    return SearchSpaceSpec(kind="float", low=low, high=high, step=step, log=log)


def _parse_search_space(payload: Mapping[str, Any], *, config_path: Path) -> dict[str, SearchSpaceSpec]:
    search_space: dict[str, SearchSpaceSpec] = {}
    for path, raw_spec in payload.items():
        if not isinstance(path, str):
            raise TypeError(f"study config {config_path} search_space keys must be strings, got {type(path)!r}")
        _validate_search_path(path, config_path=config_path)
        spec = _require_mapping(raw_spec, f"search_space.{path}", config_path=config_path)
        kind = _require_str(spec.get("type"), f"search_space.{path}.type", config_path=config_path)
        if kind == "categorical":
            search_space[path] = _parse_categorical_spec(spec, config_path=config_path, path=path)
        elif kind == "int":
            search_space[path] = _parse_int_spec(spec, config_path=config_path, path=path)
        elif kind == "float":
            search_space[path] = _parse_float_spec(spec, config_path=config_path, path=path)
        else:
            raise ValueError(f"study config {config_path} search_space '{path}' has unsupported type '{kind}'")
    if not search_space:
        raise ValueError(f"study config {config_path} field 'search_space' must not be empty")
    return search_space


def load_study_config(path: str | Path) -> StudyConfig:
    config_path = _resolve_input_path(path)
    if not config_path.exists():
        raise ValueError(f"study config file {config_path} does not exist")
    payload = _load_yaml_mapping(config_path)
    return _study_config_from_mapping(payload, source_path=config_path)
