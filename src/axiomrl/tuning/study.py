from __future__ import annotations

import csv
import io
import itertools
import json
import math
import random
import statistics
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rl_training.cli_config import serialize_train_config
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.tuning.config import SearchSpaceSpec, StudyConfig, deserialize_study_config, serialize_study_config


@dataclass(frozen=True, slots=True)
class StudyResult:
    study_dir: Path
    best_trial_index: int | None
    best_objective_value: float | None
    best_run_dir: Path | None
    best_checkpoint_path: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discrete_values(spec: SearchSpaceSpec) -> list[object]:
    if spec.kind == "categorical":
        return list(spec.values)
    if spec.kind == "int":
        assert isinstance(spec.low, int)
        assert isinstance(spec.high, int)
        step = 1 if spec.step is None else int(spec.step)
        return list(range(int(spec.low), int(spec.high) + 1, step))
    if spec.kind == "float":
        if spec.step is None:
            raise ValueError("float grid search requires an explicit step")
        assert isinstance(spec.low, float)
        assert isinstance(spec.high, float)
        step = float(spec.step)
        values: list[float] = []
        current = float(spec.low)
        while current <= float(spec.high) + (step / 10.0):
            values.append(round(current, 10))
            current += step
        return values
    raise ValueError(f"unsupported search space kind for grid search: {spec.kind}")


def _sample_random_value(spec: SearchSpaceSpec, *, rng: random.Random) -> object:
    if spec.kind == "categorical":
        return rng.choice(spec.values)
    if spec.kind == "int":
        choices = _discrete_values(spec)
        return rng.choice(choices)
    if spec.kind == "float":
        assert spec.low is not None
        assert spec.high is not None
        low = float(spec.low)
        high = float(spec.high)
        if spec.log:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        if spec.step is not None:
            step = float(spec.step)
            offset = round((value - low) / step)
            value = low + (offset * step)
            value = max(low, min(high, value))
        return round(value, 10)
    raise ValueError(f"unsupported search space kind for random search: {spec.kind}")


def _iter_trial_params(config: StudyConfig) -> list[dict[str, object]]:
    if config.study.backend != "native":
        raise ModuleNotFoundError(
            "Optuna backend is not available yet; install optional tuning support before using study.backend=optuna"
        )

    if config.study.sampler == "grid":
        search_items = list(config.search_space.items())
        candidate_values = [_discrete_values(spec) for _, spec in search_items]
        return [
            {path: value for (path, _), value in zip(search_items, combination, strict=True)}
            for combination in itertools.product(*candidate_values)
        ]

    rng = random.Random(config.study.seed)
    assert config.study.num_trials is not None
    trial_params: list[dict[str, object]] = []
    for _ in range(config.study.num_trials):
        sampled: dict[str, object] = {}
        for path, spec in config.search_space.items():
            sampled[path] = _sample_random_value(spec, rng=rng)
        trial_params.append(sampled)
    return trial_params


def _set_nested_mapping_value(mapping: dict[str, Any], path_tokens: list[str], value: object) -> None:
    current = mapping
    for token in path_tokens[:-1]:
        nested = current.get(token)
        if not isinstance(nested, dict):
            nested = {}
            current[token] = nested
        current = nested
    current[path_tokens[-1]] = value


def _apply_trial_params(base_config: TrainConfig, params: dict[str, object], *, trial_output_dir: Path) -> TrainConfig:
    config = replace(
        base_config,
        output_dir=trial_output_dir,
        algo_kwargs=deepcopy(base_config.algo_kwargs),
        env_kwargs=deepcopy(base_config.env_kwargs),
        benchmark=deepcopy(base_config.benchmark),
    )
    top_level_overrides: dict[str, object] = {}
    for path, value in params.items():
        if path.startswith("algo_kwargs."):
            _set_nested_mapping_value(config.algo_kwargs, path.split(".")[1:], value)
        elif path.startswith("env_kwargs."):
            _set_nested_mapping_value(config.env_kwargs, path.split(".")[1:], value)
        elif path.startswith("benchmark."):
            _set_nested_mapping_value(config.benchmark, path.split(".")[1:], value)
        else:
            top_level_overrides[path] = value
    if top_level_overrides:
        config = replace(config, **top_level_overrides)
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_trial_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_trial_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _is_better(candidate: float, incumbent: float | None, *, mode: str) -> bool:
    if incumbent is None:
        return True
    if mode == "min":
        return candidate < incumbent
    return candidate > incumbent


def _study_result_from_payload(study_dir: Path, payload: dict[str, Any]) -> StudyResult:
    best_trial_index = payload.get("best_trial_index")
    best_objective_value = payload.get("best_objective_value")
    best_run_dir = payload.get("best_run_dir")
    best_checkpoint_path = payload.get("best_checkpoint_path")
    return StudyResult(
        study_dir=study_dir,
        best_trial_index=None if best_trial_index is None else int(best_trial_index),
        best_objective_value=None if best_objective_value is None else float(best_objective_value),
        best_run_dir=None if best_run_dir is None else Path(best_run_dir),
        best_checkpoint_path=None if best_checkpoint_path is None else Path(best_checkpoint_path),
    )


def _load_study_summary_payload(study_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved_study_dir = Path(study_dir).resolve()
    study_json_path = resolved_study_dir / "study.json"
    if not study_json_path.exists():
        raise ValueError(f"study directory {resolved_study_dir} does not contain study.json")

    payload = json.loads(study_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {study_json_path}, got {type(payload)!r}")
    return resolved_study_dir, payload


def load_study_report(study_dir: str | Path) -> dict[str, Any]:
    resolved_study_dir, payload = _load_study_summary_payload(study_dir)
    report = dict(payload)
    report["study_dir"] = str(resolved_study_dir)
    report["trials"] = _load_trial_records(resolved_study_dir / "trials.jsonl")
    return report


def _trial_index_key(trial: Mapping[str, Any]) -> int:
    trial_index = trial.get("trial_index")
    if trial_index is None:
        return -1
    return int(trial_index)


def _status_counts_for_trials(trials: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in trials:
        status = str(trial.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _trial_matches_param_filters(trial: Mapping[str, Any], param_filters: Mapping[str, object]) -> bool:
    params = trial.get("params", {})
    if not isinstance(params, Mapping):
        return False
    for name, expected_value in param_filters.items():
        if params.get(name) != expected_value:
            return False
    return True


def _trial_matches_error_contains(trial: Mapping[str, Any], error_contains: str) -> bool:
    raw_error = trial.get("error")
    if raw_error is None:
        return False
    return error_contains.casefold() in str(raw_error).casefold()


def _trial_matches_error(trial: Mapping[str, Any], error: str) -> bool:
    raw_error = trial.get("error")
    if raw_error is None:
        return False
    return str(raw_error).strip() == error


def _normalized_error_text(raw_error: object) -> str:
    error_text = "unknown" if raw_error is None else str(raw_error).strip()
    return "unknown" if not error_text else error_text


def _error_type_from_text(error_text: str) -> str:
    first_line = error_text.splitlines()[0].strip()
    if not first_line:
        return "unknown"
    if ":" in first_line:
        error_type, _ = first_line.split(":", 1)
        normalized_error_type = error_type.strip()
        return "unknown" if not normalized_error_type else normalized_error_type
    return first_line


def _trial_matches_error_type(trial: Mapping[str, Any], error_type: str) -> bool:
    raw_error = trial.get("error")
    if raw_error is None:
        return False
    normalized_error_type = _error_type_from_text(_normalized_error_text(raw_error))
    return normalized_error_type.casefold() == error_type.casefold()


def _normalized_numeric_threshold(value: object, *, flag_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{flag_name} must be numeric")
    normalized_value = float(value)
    if math.isnan(normalized_value):
        raise ValueError(f"{flag_name} must not be NaN")
    return normalized_value


def _trial_matches_objective_thresholds(
    trial: Mapping[str, Any],
    *,
    objective_at_least: float | None,
    objective_at_most: float | None,
) -> bool:
    if objective_at_least is None and objective_at_most is None:
        return True
    raw_objective_value = trial.get("objective_value")
    if raw_objective_value is None:
        return False
    objective_value = float(raw_objective_value)
    if objective_at_least is not None and objective_value < objective_at_least:
        return False
    if objective_at_most is not None and objective_value > objective_at_most:
        return False
    return True


def _trial_matches_duration_thresholds(
    trial: Mapping[str, Any],
    *,
    duration_at_least: float | None,
    duration_at_most: float | None,
) -> bool:
    if duration_at_least is None and duration_at_most is None:
        return True
    duration_seconds = _visible_trial_duration_seconds(trial)
    if duration_seconds is None:
        return False
    if duration_at_least is not None and duration_seconds < duration_at_least:
        return False
    if duration_at_most is not None and duration_seconds > duration_at_most:
        return False
    return True


def _value_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int | float):
        return (1, float(value))
    if isinstance(value, str):
        return (2, value)
    return (3, json.dumps(value, sort_keys=True, ensure_ascii=False))


def _sorted_unique_values(values: list[object]) -> list[object]:
    unique_values: dict[str, object] = {}
    for value in values:
        unique_values[json.dumps(value, sort_keys=True, ensure_ascii=False)] = value
    return sorted(unique_values.values(), key=_value_sort_key)


def _study_report_search_space_specs(payload: Mapping[str, Any]) -> dict[str, SearchSpaceSpec]:
    study_config_payload = payload.get("study_config")
    if not isinstance(study_config_payload, Mapping):
        return {}
    search_space_payload = study_config_payload.get("search_space")
    if not isinstance(search_space_payload, Mapping):
        return {}

    specs: dict[str, SearchSpaceSpec] = {}
    for path, raw_spec in search_space_payload.items():
        if not isinstance(path, str) or not isinstance(raw_spec, Mapping):
            continue
        kind = raw_spec.get("type")
        if not isinstance(kind, str):
            continue
        values = raw_spec.get("values", ())
        if isinstance(values, tuple):
            serialized_values = values
        elif isinstance(values, list):
            serialized_values = tuple(values)
        else:
            serialized_values = ()
        specs[path] = SearchSpaceSpec(
            kind=kind,
            low=raw_spec.get("low"),
            high=raw_spec.get("high"),
            step=raw_spec.get("step"),
            log=bool(raw_spec.get("log", False)),
            values=serialized_values,
        )
    return specs


def _trial_config_path(trial: Mapping[str, Any]) -> Path | None:
    run_dir = trial.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        return None
    return config_path


def _trial_config_payload(trial: Mapping[str, Any]) -> dict[str, Any] | None:
    config_path = _trial_config_path(trial)
    if config_path is None:
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {config_path}, got {type(payload)!r}")
    return payload


def _selected_objective_summary(trials: list[dict[str, Any]]) -> dict[str, int | float | None]:
    completed_values = [
        float(trial["objective_value"])
        for trial in trials
        if trial.get("status") == "completed" and trial.get("objective_value") is not None
    ]
    failed_trials = sum(1 for trial in trials if trial.get("status") == "failed")
    if not completed_values:
        return {
            "completed_trials": 0,
            "failed_trials": failed_trials,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "completed_trials": len(completed_values),
        "failed_trials": failed_trials,
        "min": min(completed_values),
        "max": max(completed_values),
        "mean": statistics.fmean(completed_values),
        "median": statistics.median(completed_values),
    }


def _selected_error_summaries(trials: list[dict[str, Any]]) -> list[dict[str, object]]:
    failed_trials = [trial for trial in trials if trial.get("status") == "failed"]
    if not failed_trials:
        return []
    grouped_trial_indices: dict[str, list[int]] = {}
    for trial in failed_trials:
        error_text = _normalized_error_text(trial.get("error"))
        grouped_trial_indices.setdefault(error_text, []).append(int(trial["trial_index"]))
    selected_trial_count = len(trials)
    failed_trial_count = len(failed_trials)
    summaries: list[dict[str, object]] = []
    for error_text, trial_indices in grouped_trial_indices.items():
        normalized_indices = sorted(trial_indices)
        count = len(normalized_indices)
        summaries.append(
            {
                "error": error_text,
                "failed_trials": count,
                "selected_trial_share": round(count / selected_trial_count, 10),
                "failed_trial_share": round(count / failed_trial_count, 10),
                "trial_indices": normalized_indices,
            }
        )
    summaries.sort(key=lambda entry: (-int(entry["failed_trials"]), str(entry["error"])))
    return summaries


def _selected_error_type_summaries(trials: list[dict[str, Any]]) -> list[dict[str, object]]:
    failed_trials = [trial for trial in trials if trial.get("status") == "failed"]
    if not failed_trials:
        return []
    grouped_entries: dict[str, dict[str, object]] = {}
    for trial in failed_trials:
        error_text = _normalized_error_text(trial.get("error"))
        error_type = _error_type_from_text(error_text)
        grouped_entry = grouped_entries.setdefault(
            error_type,
            {
                "error_type": error_type,
                "errors": set(),
                "trial_indices": [],
            },
        )
        assert isinstance(grouped_entry["errors"], set)
        assert isinstance(grouped_entry["trial_indices"], list)
        grouped_entry["errors"].add(error_text)
        grouped_entry["trial_indices"].append(int(trial["trial_index"]))
    selected_trial_count = len(trials)
    failed_trial_count = len(failed_trials)
    summaries: list[dict[str, object]] = []
    for grouped_entry in grouped_entries.values():
        trial_indices = sorted(int(index) for index in grouped_entry["trial_indices"])
        failed_count = len(trial_indices)
        summaries.append(
            {
                "error_type": grouped_entry["error_type"],
                "errors": sorted(str(error_text) for error_text in grouped_entry["errors"]),
                "failed_trials": failed_count,
                "selected_trial_share": round(failed_count / selected_trial_count, 10),
                "failed_trial_share": round(failed_count / failed_trial_count, 10),
                "trial_indices": trial_indices,
            }
        )
    summaries.sort(key=lambda entry: (-int(entry["failed_trials"]), str(entry["error_type"])))
    return summaries


def _parse_study_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _trial_duration_seconds(trial: Mapping[str, Any]) -> float | None:
    started_at = _parse_study_timestamp(trial.get("started_at"))
    ended_at = _parse_study_timestamp(trial.get("ended_at"))
    if started_at is None or ended_at is None:
        return None
    return _timestamp_delta_seconds(ended_at, started_at)


def _timestamp_delta_seconds(later: datetime, earlier: datetime) -> float | None:
    duration_seconds = (later - earlier).total_seconds()
    if duration_seconds < 0:
        return None
    return round(duration_seconds, 10)


def _trial_ended_at(trial: Mapping[str, Any]) -> datetime | None:
    return _parse_study_timestamp(trial.get("ended_at"))


def _visible_trial_age_seconds(
    *,
    trial_ended_at: datetime | None,
    incumbent_ended_at: datetime | None,
) -> float | None:
    if trial_ended_at is None or incumbent_ended_at is None:
        return None
    return _timestamp_delta_seconds(trial_ended_at, incumbent_ended_at)


def _visible_trial_age_trials(
    *,
    trial_position: int,
    incumbent_position: int | None,
) -> int | None:
    if incumbent_position is None:
        return None
    return trial_position - incumbent_position


def _visible_trial_duration_seconds(trial: Mapping[str, Any]) -> float | None:
    raw_duration = trial.get("duration_seconds")
    if isinstance(raw_duration, int | float) and not isinstance(raw_duration, bool):
        return round(float(raw_duration), 10)
    return _trial_duration_seconds(trial)


def _duration_values_for_trials(trials: list[dict[str, Any]]) -> list[float]:
    return [
        float(duration_seconds)
        for trial in trials
        for duration_seconds in [_visible_trial_duration_seconds(trial)]
        if duration_seconds is not None
    ]


def _selected_duration_summary(trials: list[dict[str, Any]]) -> dict[str, int | float | None]:
    durations = _duration_values_for_trials(trials)
    if not durations:
        return {
            "timed_trials": 0,
            "untimed_trials": len(trials),
            "min_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
        }
    return {
        "timed_trials": len(durations),
        "untimed_trials": len(trials) - len(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "mean_seconds": statistics.fmean(durations),
        "median_seconds": statistics.median(durations),
    }


def _objective_duration_dominates(
    candidate: Mapping[str, Any],
    other: Mapping[str, Any],
    *,
    mode: str,
) -> bool:
    candidate_objective_value = float(candidate["objective_value"])
    other_objective_value = float(other["objective_value"])
    candidate_duration_seconds = float(candidate["duration_seconds"])
    other_duration_seconds = float(other["duration_seconds"])
    if mode == "min":
        objective_is_no_worse = candidate_objective_value <= other_objective_value
    else:
        objective_is_no_worse = candidate_objective_value >= other_objective_value
    duration_is_no_worse = candidate_duration_seconds <= other_duration_seconds
    return objective_is_no_worse and duration_is_no_worse and (
        candidate_objective_value != other_objective_value
        or candidate_duration_seconds != other_duration_seconds
    )


def _selected_objective_duration_frontier(
    trials: list[dict[str, Any]],
    *,
    objective_mode: str,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for trial in trials:
        if trial.get("status") != "completed":
            continue
        raw_objective_value = trial.get("objective_value")
        if raw_objective_value is None:
            continue
        duration_seconds = _visible_trial_duration_seconds(trial)
        if duration_seconds is None:
            continue
        params = trial.get("params", {})
        params_mapping = params if isinstance(params, Mapping) else {}
        candidates.append(
            {
                "trial_index": _trial_index_key(trial),
                "objective_value": float(raw_objective_value),
                "duration_seconds": duration_seconds,
                "selected_best_objective_delta": trial.get("selected_best_objective_delta"),
                "params": dict(params_mapping),
            }
        )
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and _objective_duration_dominates(other, candidate, mode=objective_mode)
            for other in candidates
        )
    ]
    frontier.sort(
        key=lambda entry: (
            float(entry["duration_seconds"]),
            float(entry["objective_value"]) if objective_mode == "min" else -float(entry["objective_value"]),
            int(entry["trial_index"]),
        )
    )
    return frontier


def _objective_delta(
    objective_value: float | None,
    *,
    best_value: float | None,
    mode: str,
) -> float | None:
    if objective_value is None or best_value is None:
        return None
    if mode == "min":
        return round(objective_value - best_value, 10)
    return round(best_value - objective_value, 10)


def _incumbent_update_improvement(
    objective_value: float | None,
    *,
    previous_incumbent_objective_value: float | None,
    mode: str,
) -> float | None:
    if objective_value is None or previous_incumbent_objective_value is None:
        return None
    if mode == "min":
        return round(previous_incumbent_objective_value - objective_value, 10)
    return round(objective_value - previous_incumbent_objective_value, 10)


def _selected_incumbent_trace(
    trials: list[dict[str, Any]],
    *,
    objective_mode: str,
) -> list[dict[str, object]]:
    ordered_trials = sorted(trials, key=_trial_index_key)
    incumbent_trial_index: int | None = None
    incumbent_objective_value: float | None = None
    incumbent_position: int | None = None
    incumbent_ended_at: datetime | None = None
    previous_update_position: int | None = None
    trace: list[dict[str, object]] = []
    for position, trial in enumerate(ordered_trials):
        raw_objective_value = trial.get("objective_value")
        objective_value = None if raw_objective_value is None else float(raw_objective_value)
        trial_ended_at = _trial_ended_at(trial)
        is_incumbent_update = False
        update_improvement: float | None = None
        trials_since_previous_update: int | None = None
        if trial.get("status") == "completed" and objective_value is not None:
            if _is_better(objective_value, incumbent_objective_value, mode=objective_mode):
                update_improvement = _incumbent_update_improvement(
                    objective_value,
                    previous_incumbent_objective_value=incumbent_objective_value,
                    mode=objective_mode,
                )
                if previous_update_position is not None:
                    trials_since_previous_update = position - previous_update_position
                incumbent_trial_index = _trial_index_key(trial)
                incumbent_objective_value = objective_value
                incumbent_position = position
                incumbent_ended_at = trial_ended_at
                previous_update_position = position
                is_incumbent_update = True
        incumbent_age_trials = _visible_trial_age_trials(
            trial_position=position,
            incumbent_position=incumbent_position,
        )
        incumbent_age_seconds = _visible_trial_age_seconds(
            trial_ended_at=trial_ended_at,
            incumbent_ended_at=incumbent_ended_at,
        )
        trace.append(
            {
                "trial_index": _trial_index_key(trial),
                "status": trial.get("status"),
                "objective_value": objective_value,
                "selected_incumbent_trial_index": incumbent_trial_index,
                "selected_incumbent_objective_value": incumbent_objective_value,
                "selected_is_incumbent_update": is_incumbent_update,
                "selected_incumbent_update_improvement": update_improvement,
                "selected_incumbent_trials_since_previous_update": trials_since_previous_update,
                "selected_incumbent_age_trials": incumbent_age_trials,
                "selected_incumbent_age_seconds": incumbent_age_seconds,
            }
        )
    return trace


def _selected_incumbent_update_summary(
    trace: list[dict[str, object]],
) -> dict[str, object]:
    update_entries = [
        entry
        for entry in trace
        if entry.get("selected_is_incumbent_update") is True
    ]
    summary: dict[str, object] = {
        "incumbent_update_count": len(update_entries),
        "first_incumbent_trial_index": None,
        "latest_incumbent_trial_index": None,
        "latest_incumbent_objective_value": None,
        "mean_improvement_over_previous": None,
        "max_improvement_over_previous": None,
        "mean_trials_since_previous_update": None,
        "max_trials_since_previous_update": None,
    }
    if not update_entries:
        return summary
    summary["first_incumbent_trial_index"] = update_entries[0].get("trial_index")
    summary["latest_incumbent_trial_index"] = update_entries[-1].get("trial_index")
    summary["latest_incumbent_objective_value"] = update_entries[-1].get("selected_incumbent_objective_value")

    improvements = [
        float(value)
        for entry in update_entries
        if isinstance((value := entry.get("selected_incumbent_update_improvement")), int | float)
        and not isinstance(value, bool)
    ]
    if improvements:
        summary["mean_improvement_over_previous"] = round(statistics.fmean(improvements), 10)
        summary["max_improvement_over_previous"] = max(improvements)

    trial_gaps = [
        int(value)
        for entry in update_entries
        if isinstance((value := entry.get("selected_incumbent_trials_since_previous_update")), int)
        and not isinstance(value, bool)
    ]
    if trial_gaps:
        summary["mean_trials_since_previous_update"] = round(statistics.fmean(trial_gaps), 10)
        summary["max_trials_since_previous_update"] = max(trial_gaps)
    return summary


def _selected_incumbent_staleness_summary(
    trace: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "latest_incumbent_age_trials": None,
        "latest_incumbent_age_seconds": None,
        "max_incumbent_age_trials": None,
        "max_incumbent_age_seconds": None,
    }
    if trace:
        latest_entry = trace[-1]
        summary["latest_incumbent_age_trials"] = latest_entry.get("selected_incumbent_age_trials")
        summary["latest_incumbent_age_seconds"] = latest_entry.get("selected_incumbent_age_seconds")

    trial_ages = [
        int(value)
        for entry in trace
        if isinstance((value := entry.get("selected_incumbent_age_trials")), int)
        and not isinstance(value, bool)
    ]
    if trial_ages:
        summary["max_incumbent_age_trials"] = max(trial_ages)

    second_ages = [
        float(value)
        for entry in trace
        if isinstance((value := entry.get("selected_incumbent_age_seconds")), int | float)
        and not isinstance(value, bool)
    ]
    if second_ages:
        summary["max_incumbent_age_seconds"] = max(second_ages)
    return summary


def _selected_parameter_summaries(
    trials: list[dict[str, Any]],
    *,
    objective_mode: str,
    search_space_specs: Mapping[str, SearchSpaceSpec] | None = None,
) -> dict[str, dict[str, object]]:
    param_names: set[str] = set()
    for trial in trials:
        params = trial.get("params", {})
        if isinstance(params, Mapping):
            param_names.update(str(key) for key in params)

    best_trial, _ = _best_record_from_records(trials, mode=objective_mode)
    best_params = best_trial.get("params", {}) if isinstance(best_trial, Mapping) else {}
    best_params_mapping = best_params if isinstance(best_params, Mapping) else {}

    summaries: dict[str, dict[str, object]] = {}
    for param_name in sorted(param_names):
        completed_values = [
            trial["params"][param_name]
            for trial in trials
            if trial.get("status") == "completed"
            and isinstance(trial.get("params"), Mapping)
            and param_name in trial["params"]
        ]
        failed_values = [
            trial["params"][param_name]
            for trial in trials
            if trial.get("status") == "failed"
            and isinstance(trial.get("params"), Mapping)
            and param_name in trial["params"]
        ]
        summary: dict[str, object] = {
            "completed_unique_values": _sorted_unique_values(completed_values),
            "failed_unique_values": _sorted_unique_values(failed_values),
            "selected_best_value": best_params_mapping.get(param_name),
        }
        observed_values = _sorted_unique_values([*completed_values, *failed_values])
        summary["observed_unique_values"] = observed_values
        summary["observed_unique_count"] = len(observed_values)
        spec = None if search_space_specs is None else search_space_specs.get(param_name)
        if spec is not None:
            summary["search_space_kind"] = spec.kind
            candidate_values: list[object] | None = None
            if spec.kind == "categorical" or spec.kind == "int" or (spec.kind == "float" and spec.step is not None):
                candidate_values = _discrete_values(spec)
            if candidate_values is not None:
                summary["candidate_count"] = len(candidate_values)
                if candidate_values:
                    summary["coverage_ratio"] = round(len(observed_values) / len(candidate_values), 10)
        if completed_values and all(isinstance(value, int | float) and not isinstance(value, bool) for value in completed_values):
            numeric_values = [float(value) for value in completed_values]
            summary["numeric_min"] = min(numeric_values)
            summary["numeric_max"] = max(numeric_values)
            summary["numeric_mean"] = statistics.fmean(numeric_values)
        summaries[param_name] = summary
    return summaries


def _selected_parameter_value_summaries(
    trials: list[dict[str, Any]],
    *,
    objective_mode: str,
    selected_best_objective_value: float | None,
) -> dict[str, list[dict[str, object]]]:
    def _attach_rank_field(entries: list[dict[str, object]], *, source_field: str, rank_field: str) -> None:
        ranked_entries: list[tuple[int, float, object]] = []
        for index, entry in enumerate(entries):
            raw_value = entry.get(source_field)
            if not isinstance(raw_value, int | float) or isinstance(raw_value, bool):
                entry[rank_field] = None
                continue
            ranked_entries.append((index, float(raw_value), entry.get("value")))
        if objective_mode == "min":
            ranked_entries.sort(key=lambda item: (item[1], _value_sort_key(item[2])))
        else:
            ranked_entries.sort(key=lambda item: (-item[1], _value_sort_key(item[2])))
        for rank, (index, _, _) in enumerate(ranked_entries, start=1):
            entries[index][rank_field] = rank

    param_names: set[str] = set()
    for trial in trials:
        params = trial.get("params", {})
        if isinstance(params, Mapping):
            param_names.update(str(key) for key in params)

    summaries: dict[str, list[dict[str, object]]] = {}
    for param_name in sorted(param_names):
        grouped_trials: dict[str, list[dict[str, Any]]] = {}
        grouped_values: dict[str, object] = {}
        for trial in trials:
            params = trial.get("params", {})
            if not isinstance(params, Mapping) or param_name not in params:
                continue
            value = params[param_name]
            value_key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            grouped_values[value_key] = value
            grouped_trials.setdefault(value_key, []).append(trial)

        value_summaries: list[dict[str, object]] = []
        for value in _sorted_unique_values(list(grouped_values.values())):
            value_key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            value_trials = grouped_trials.get(value_key, [])
            completed_trials = [
                trial
                for trial in value_trials
                if trial.get("status") == "completed"
            ]
            failed_trials = [
                trial
                for trial in value_trials
                if trial.get("status") == "failed"
            ]
            completed_objective_values = [
                float(trial["objective_value"])
                for trial in completed_trials
                if trial.get("objective_value") is not None
            ]
            incumbent_update_trial_indices = sorted(
                _trial_index_key(trial)
                for trial in value_trials
                if bool(trial.get("selected_is_incumbent_update"))
            )
            durations = _duration_values_for_trials(value_trials)
            _, best_objective_value = _best_record_from_records(value_trials, mode=objective_mode)
            value_summaries.append(
                {
                    "value": value,
                    "trial_count": len(value_trials),
                    "completed_trials": len(completed_trials),
                    "failed_trials": len(failed_trials),
                    "timed_trials": len(durations),
                    "untimed_trials": len(value_trials) - len(durations),
                    "completion_rate": (
                        None if not value_trials else round(len(completed_trials) / len(value_trials), 10)
                    ),
                    "failure_rate": (
                        None if not value_trials else round(len(failed_trials) / len(value_trials), 10)
                    ),
                    "best_objective_value": best_objective_value,
                    "mean_objective_value": (
                        None if not completed_objective_values else statistics.fmean(completed_objective_values)
                    ),
                    "median_objective_value": (
                        None if not completed_objective_values else statistics.median(completed_objective_values)
                    ),
                    "min_duration_seconds": None if not durations else min(durations),
                    "max_duration_seconds": None if not durations else max(durations),
                    "mean_duration_seconds": None if not durations else statistics.fmean(durations),
                    "median_duration_seconds": None if not durations else statistics.median(durations),
                    "incumbent_updates": len(incumbent_update_trial_indices),
                    "latest_incumbent_trial_index": (
                        None if not incumbent_update_trial_indices else max(incumbent_update_trial_indices)
                    ),
                    "selected_best_objective_delta": _objective_delta(
                        best_objective_value,
                        best_value=selected_best_objective_value,
                        mode=objective_mode,
                    ),
                }
            )
        _attach_rank_field(
            value_summaries,
            source_field="best_objective_value",
            rank_field="rank_by_best_objective_value",
        )
        _attach_rank_field(
            value_summaries,
            source_field="mean_objective_value",
            rank_field="rank_by_mean_objective_value",
        )
        summaries[param_name] = value_summaries
    return summaries


def _selected_parameter_incumbent_summaries(
    parameter_value_summaries: Mapping[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for param_name, entries in parameter_value_summaries.items():
        contributing_entries = [
            entry
            for entry in entries
            if isinstance(entry.get("incumbent_updates"), int)
            and int(entry["incumbent_updates"]) > 0
        ]
        contributing_values = _sorted_unique_values([entry.get("value") for entry in contributing_entries])
        incumbent_update_count = sum(int(entry["incumbent_updates"]) for entry in contributing_entries)
        top_entry: Mapping[str, object] | None = None
        latest_entry: Mapping[str, object] | None = None
        if contributing_entries:
            top_entry = min(
                contributing_entries,
                key=lambda entry: (
                    -int(entry["incumbent_updates"]),
                    (
                        0
                        if isinstance(entry.get("latest_incumbent_trial_index"), int)
                        else 1
                    ),
                    (
                        0
                        if not isinstance(entry.get("latest_incumbent_trial_index"), int)
                        else -int(entry["latest_incumbent_trial_index"])
                    ),
                    _value_sort_key(entry.get("value")),
                ),
            )
            latest_entry = min(
                contributing_entries,
                key=lambda entry: (
                    (
                        0
                        if isinstance(entry.get("latest_incumbent_trial_index"), int)
                        else 1
                    ),
                    (
                        0
                        if not isinstance(entry.get("latest_incumbent_trial_index"), int)
                        else -int(entry["latest_incumbent_trial_index"])
                    ),
                    _value_sort_key(entry.get("value")),
                ),
            )
        summaries[str(param_name)] = {
            "incumbent_update_count": incumbent_update_count,
            "contributing_values": contributing_values,
            "contributing_value_count": len(contributing_values),
            "top_incumbent_value": None if top_entry is None else top_entry.get("value"),
            "top_incumbent_value_updates": 0 if top_entry is None else int(top_entry["incumbent_updates"]),
            "latest_incumbent_value": None if latest_entry is None else latest_entry.get("value"),
            "latest_incumbent_trial_index": (
                None if latest_entry is None else latest_entry.get("latest_incumbent_trial_index")
            ),
        }
    return summaries


def _selected_parameter_incumbent_leaderboard(
    parameter_incumbent_summaries: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for param_name, summary in parameter_incumbent_summaries.items():
        entries.append(
            {
                "name": str(param_name),
                "incumbent_update_count": summary.get("incumbent_update_count"),
                "contributing_values": summary.get("contributing_values"),
                "contributing_value_count": summary.get("contributing_value_count"),
                "top_incumbent_value": summary.get("top_incumbent_value"),
                "top_incumbent_value_updates": summary.get("top_incumbent_value_updates"),
                "latest_incumbent_value": summary.get("latest_incumbent_value"),
                "latest_incumbent_trial_index": summary.get("latest_incumbent_trial_index"),
            }
        )
    entries.sort(
        key=lambda entry: (
            -int(entry["incumbent_update_count"]),
            -int(entry["contributing_value_count"]),
            (
                1
                if entry.get("latest_incumbent_trial_index") is None
                else -int(entry["latest_incumbent_trial_index"])
            ),
            str(entry["name"]),
        )
    )
    return entries


def _selected_parameter_effect_leaderboard(
    parameter_value_summaries: Mapping[str, list[dict[str, object]]],
    *,
    objective_mode: str,
) -> list[dict[str, object]]:
    def _extremes_for_field(
        entries: list[dict[str, object]],
        *,
        field: str,
    ) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None, float | None]:
        scored_entries = [
            entry
            for entry in entries
            if isinstance(entry.get(field), int | float)
            and not isinstance(entry.get(field), bool)
        ]
        if not scored_entries:
            return None, None, None
        ranked_entries = sorted(
            scored_entries,
            key=lambda entry: (
                float(entry[field]) if objective_mode == "min" else -float(entry[field]),
                _value_sort_key(entry.get("value")),
            ),
        )
        top_entry = ranked_entries[0]
        bottom_entry = ranked_entries[-1]
        spread = round(abs(float(top_entry[field]) - float(bottom_entry[field])), 10)
        return top_entry, bottom_entry, spread

    leaderboard: list[dict[str, object]] = []
    for param_name, entries in parameter_value_summaries.items():
        observed_value_count = len(entries)
        completed_value_count = sum(
            1
            for entry in entries
            if isinstance(entry.get("completed_trials"), int)
            and int(entry["completed_trials"]) > 0
        )
        top_best_entry, bottom_best_entry, best_spread = _extremes_for_field(
            entries,
            field="best_objective_value",
        )
        top_mean_entry, bottom_mean_entry, mean_spread = _extremes_for_field(
            entries,
            field="mean_objective_value",
        )
        leaderboard.append(
            {
                "name": str(param_name),
                "observed_value_count": observed_value_count,
                "completed_value_count": completed_value_count,
                "best_objective_spread": best_spread,
                "mean_objective_spread": mean_spread,
                "top_value_by_best_objective": None if top_best_entry is None else top_best_entry.get("value"),
                "top_best_objective_value": (
                    None if top_best_entry is None else top_best_entry.get("best_objective_value")
                ),
                "bottom_value_by_best_objective": (
                    None if bottom_best_entry is None else bottom_best_entry.get("value")
                ),
                "bottom_best_objective_value": (
                    None if bottom_best_entry is None else bottom_best_entry.get("best_objective_value")
                ),
                "top_value_by_mean_objective": None if top_mean_entry is None else top_mean_entry.get("value"),
                "top_mean_objective_value": (
                    None if top_mean_entry is None else top_mean_entry.get("mean_objective_value")
                ),
                "bottom_value_by_mean_objective": (
                    None if bottom_mean_entry is None else bottom_mean_entry.get("value")
                ),
                "bottom_mean_objective_value": (
                    None if bottom_mean_entry is None else bottom_mean_entry.get("mean_objective_value")
                ),
            }
        )
    leaderboard.sort(
        key=lambda entry: (
            entry.get("best_objective_spread") is None,
            0.0 if entry.get("best_objective_spread") is None else -float(entry["best_objective_spread"]),
            entry.get("mean_objective_spread") is None,
            0.0 if entry.get("mean_objective_spread") is None else -float(entry["mean_objective_spread"]),
            -int(entry["completed_value_count"]),
            -int(entry["observed_value_count"]),
            str(entry["name"]),
        )
    )
    return leaderboard


def _focused_parameter_value_summary(
    parameter_value_summaries: Mapping[str, list[dict[str, object]]],
    *,
    focus_param: str,
    focus_sort_by: str = "best-objective-value",
    focus_top_k: int | None = None,
) -> list[dict[str, object]]:
    if focus_sort_by not in {
        "best-objective-value",
        "mean-objective-value",
        "completion-rate",
        "incumbent-updates",
        "mean-duration-seconds",
        "value",
    }:
        raise ValueError(f"unsupported focus parameter sort field: {focus_sort_by}")
    if focus_top_k is not None and focus_top_k < 1:
        raise ValueError("--focus-top-k must be greater than or equal to 1")
    focused_entries = parameter_value_summaries.get(focus_param)
    if focused_entries is None:
        raise ValueError(f"--focus-param {focus_param!r} was not found in the visible trial slice")
    sorted_entries = [dict(entry) for entry in focused_entries]
    if focus_sort_by == "value":
        sorted_entries.sort(key=lambda entry: _value_sort_key(entry.get("value")))
        return sorted_entries
    if focus_sort_by == "completion-rate":
        sorted_entries.sort(
            key=lambda entry: (
                entry.get("completion_rate") is None,
                float("inf") if entry.get("completion_rate") is None else -float(entry["completion_rate"]),
                _value_sort_key(entry.get("value")),
            )
        )
    elif focus_sort_by == "incumbent-updates":
        sorted_entries.sort(
            key=lambda entry: (
                entry.get("incumbent_updates") is None,
                float("inf") if entry.get("incumbent_updates") is None else -float(entry["incumbent_updates"]),
                entry.get("latest_incumbent_trial_index") is None,
                (
                    float("inf")
                    if entry.get("latest_incumbent_trial_index") is None
                    else -float(entry["latest_incumbent_trial_index"])
                ),
                _value_sort_key(entry.get("value")),
            )
        )
    elif focus_sort_by == "mean-duration-seconds":
        sorted_entries.sort(
            key=lambda entry: (
                entry.get("mean_duration_seconds") is None,
                float("inf") if entry.get("mean_duration_seconds") is None else float(entry["mean_duration_seconds"]),
                _value_sort_key(entry.get("value")),
            )
        )
    else:
        rank_field = (
            "rank_by_best_objective_value"
            if focus_sort_by == "best-objective-value"
            else "rank_by_mean_objective_value"
        )
        sorted_entries.sort(
            key=lambda entry: (
                entry.get(rank_field) is None,
                float("inf") if entry.get(rank_field) is None else int(entry[rank_field]),
                _value_sort_key(entry.get("value")),
            )
        )
    if focus_top_k is not None:
        sorted_entries = sorted_entries[:focus_top_k]
    return sorted_entries


def _coverage_summary_entry(
    parameter_summaries: Mapping[str, Mapping[str, Any]],
    *,
    pick: str,
) -> dict[str, object] | None:
    candidates: list[tuple[float, str, int, int]] = []
    for param_name, summary in parameter_summaries.items():
        coverage_ratio = summary.get("coverage_ratio")
        candidate_count = summary.get("candidate_count")
        observed_unique_count = summary.get("observed_unique_count")
        if not isinstance(coverage_ratio, int | float):
            continue
        if not isinstance(candidate_count, int):
            continue
        if not isinstance(observed_unique_count, int):
            continue
        candidates.append((float(coverage_ratio), str(param_name), candidate_count, observed_unique_count))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[0] if pick == "lowest" else candidates[-1]
    coverage_ratio, param_name, candidate_count, observed_unique_count = selected
    return {
        "name": param_name,
        "coverage_ratio": coverage_ratio,
        "candidate_count": candidate_count,
        "observed_unique_count": observed_unique_count,
    }


def _search_progress_summary(
    trials: list[dict[str, Any]],
    *,
    selected_best_trial_index: int | None,
    completed_trials: int,
) -> dict[str, int | float | None]:
    if selected_best_trial_index is None:
        return {
            "selected_trials_until_best": None,
            "selected_trial_share_until_best": None,
            "completed_trials_until_best": None,
            "completed_trial_share_until_best": None,
            "time_to_best_seconds": None,
        }
    ordered_trials = sorted(trials, key=_trial_index_key)
    best_position = None
    best_trial: Mapping[str, Any] | None = None
    for index, trial in enumerate(ordered_trials, start=1):
        if _trial_index_key(trial) == selected_best_trial_index:
            best_position = index
            best_trial = trial
            break
    if best_position is None or best_trial is None:
        return {
            "selected_trials_until_best": None,
            "selected_trial_share_until_best": None,
            "completed_trials_until_best": None,
            "completed_trial_share_until_best": None,
            "time_to_best_seconds": None,
        }
    visible_prefix = ordered_trials[:best_position]
    completed_trials_until_best = sum(1 for trial in visible_prefix if trial.get("status") == "completed")
    started_timestamps = [
        started_at
        for trial in ordered_trials
        for started_at in [_parse_study_timestamp(trial.get("started_at"))]
        if started_at is not None
    ]
    best_ended_at = _parse_study_timestamp(best_trial.get("ended_at"))
    time_to_best_seconds = None
    if started_timestamps and best_ended_at is not None:
        earliest_started_at = min(started_timestamps)
        elapsed_seconds = (best_ended_at - earliest_started_at).total_seconds()
        if elapsed_seconds >= 0:
            time_to_best_seconds = round(elapsed_seconds, 10)
    return {
        "selected_trials_until_best": best_position,
        "selected_trial_share_until_best": round(best_position / len(ordered_trials), 10),
        "completed_trials_until_best": completed_trials_until_best,
        "completed_trial_share_until_best": (
            None if completed_trials == 0 else round(completed_trials_until_best / completed_trials, 10)
        ),
        "time_to_best_seconds": time_to_best_seconds,
    }


def _search_efficiency_summary(
    *,
    trials: list[dict[str, Any]],
    selected_trial_count: int,
    selected_best_trial_index: int | None,
    selected_best_objective_value: float | None,
    selected_objective_summary: Mapping[str, Any],
    selected_parameter_summaries: Mapping[str, Mapping[str, Any]],
    objective_mode: str,
) -> dict[str, object]:
    completed_trials = int(selected_objective_summary.get("completed_trials", 0))
    failed_trials = int(selected_objective_summary.get("failed_trials", 0))
    mean_value = selected_objective_summary.get("mean")
    median_value = selected_objective_summary.get("median")
    mean_number = float(mean_value) if isinstance(mean_value, int | float) else None
    median_number = float(median_value) if isinstance(median_value, int | float) else None
    failure_rate = None if selected_trial_count == 0 else round(failed_trials / selected_trial_count, 10)
    progress_summary = _search_progress_summary(
        trials,
        selected_best_trial_index=selected_best_trial_index,
        completed_trials=completed_trials,
    )
    return {
        "selected_trial_count": selected_trial_count,
        "completed_trials": completed_trials,
        "failed_trials": failed_trials,
        "failure_rate": failure_rate,
        "selected_best_trial_index": selected_best_trial_index,
        "selected_best_objective_value": selected_best_objective_value,
        **progress_summary,
        "best_vs_median_delta": _objective_delta(
            median_number,
            best_value=selected_best_objective_value,
            mode=objective_mode,
        ),
        "best_vs_mean_delta": _objective_delta(
            mean_number,
            best_value=selected_best_objective_value,
            mode=objective_mode,
        ),
        "lowest_coverage_parameter": _coverage_summary_entry(selected_parameter_summaries, pick="lowest"),
        "highest_coverage_parameter": _coverage_summary_entry(selected_parameter_summaries, pick="highest"),
    }


def export_selected_study_configs(payload: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    exported_trials: list[dict[str, object]] = []
    skipped_trial_indices: list[int] = []

    trials = payload.get("trials", [])
    if isinstance(trials, list):
        export_rank = 0
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            if str(trial.get("status")) != "completed":
                continue
            trial_index = trial.get("trial_index")
            if not isinstance(trial_index, int):
                skipped_trial_indices.append(-1)
                continue
            config_payload = _trial_config_payload(trial)
            if config_payload is None:
                skipped_trial_indices.append(trial_index)
                continue
            export_rank += 1
            rank = export_rank
            exported_config_path = resolved_output_dir / f"rank-{rank:03d}_trial-{trial_index:04d}.yaml"
            exported_config_path.write_text(
                yaml.safe_dump(config_payload, sort_keys=False),
                encoding="utf-8",
            )
            source_config_path = _trial_config_path(trial)
            exported_trials.append(
                {
                    "rank": rank,
                    "trial_index": trial_index,
                    "objective_value": trial.get("objective_value"),
                    "run_dir": trial.get("run_dir"),
                    "checkpoint_path": trial.get("checkpoint_path"),
                    "source_config_path": None if source_config_path is None else str(source_config_path),
                    "exported_config_path": str(exported_config_path),
                }
            )

    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "study_name": payload.get("study_name"),
                "study_dir": payload.get("study_dir"),
                "exported_trials": exported_trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(resolved_output_dir),
        "manifest_path": str(manifest_path),
        "exported_count": len(exported_trials),
        "skipped_trial_indices": skipped_trial_indices,
    }


def select_study_report(
    payload: Mapping[str, Any],
    *,
    status: str = "all",
    sort_by: str = "trial-index",
    descending: bool = False,
    top_k: int | None = None,
    frontier_only: bool = False,
    objective_at_least: float | None = None,
    objective_at_most: float | None = None,
    duration_at_least: float | None = None,
    duration_at_most: float | None = None,
    param_filters: Mapping[str, object] | None = None,
    error: str | None = None,
    error_contains: str | None = None,
    error_type: str | None = None,
    focus_param: str | None = None,
    focus_sort_by: str | None = None,
    focus_top_k: int | None = None,
) -> dict[str, Any]:
    if status not in {"all", "completed", "failed"}:
        raise ValueError(f"unsupported study report status filter: {status}")
    if sort_by not in {"trial-index", "objective-value", "duration-seconds"}:
        raise ValueError(f"unsupported study report sort field: {sort_by}")
    if top_k is not None and top_k < 1:
        raise ValueError("--top-k must be greater than or equal to 1")
    if focus_sort_by is not None and focus_param is None:
        raise ValueError("--focus-sort-by requires --focus-param")
    if focus_top_k is not None and focus_param is None:
        raise ValueError("--focus-top-k requires --focus-param")
    if focus_top_k is not None and focus_top_k < 1:
        raise ValueError("--focus-top-k must be greater than or equal to 1")
    normalized_objective_at_least = _normalized_numeric_threshold(
        objective_at_least,
        flag_name="--objective-at-least",
    )
    normalized_objective_at_most = _normalized_numeric_threshold(
        objective_at_most,
        flag_name="--objective-at-most",
    )
    normalized_duration_at_least = _normalized_numeric_threshold(
        duration_at_least,
        flag_name="--duration-at-least",
    )
    normalized_duration_at_most = _normalized_numeric_threshold(
        duration_at_most,
        flag_name="--duration-at-most",
    )
    if (
        normalized_objective_at_least is not None
        and normalized_objective_at_most is not None
        and normalized_objective_at_least > normalized_objective_at_most
    ):
        raise ValueError("--objective-at-least cannot be greater than --objective-at-most")
    if (
        normalized_duration_at_least is not None
        and normalized_duration_at_most is not None
        and normalized_duration_at_least > normalized_duration_at_most
    ):
        raise ValueError("--duration-at-least cannot be greater than --duration-at-most")
    normalized_param_filters = None if not param_filters else {str(key): value for key, value in param_filters.items()}
    normalized_error = None if error is None else str(error).strip()
    normalized_error_contains = None if error_contains is None else str(error_contains).strip()
    normalized_error_type = None if error_type is None else str(error_type).strip()
    if normalized_error == "":
        raise ValueError("--error must not be empty")
    if normalized_error_contains == "":
        raise ValueError("--error-contains must not be empty")
    if normalized_error_type == "":
        raise ValueError("--error-type must not be empty")
    if normalized_error is not None and normalized_error_contains is not None:
        raise ValueError("--error and --error-contains cannot be used together")

    objective = payload.get("objective", {})
    objective_mapping = objective if isinstance(objective, Mapping) else {}
    objective_mode = str(objective_mapping.get("mode", "max"))
    trials = payload.get("trials", [])
    selected_trials = [
        dict(trial)
        for trial in trials
        if isinstance(trial, Mapping)
        and (status == "all" or str(trial.get("status")) == status)
        and _trial_matches_objective_thresholds(
            trial,
            objective_at_least=normalized_objective_at_least,
            objective_at_most=normalized_objective_at_most,
        )
        and _trial_matches_duration_thresholds(
            trial,
            duration_at_least=normalized_duration_at_least,
            duration_at_most=normalized_duration_at_most,
        )
        and (
            normalized_param_filters is None
            or _trial_matches_param_filters(trial, normalized_param_filters)
        )
        and (
            normalized_error is None
            or _trial_matches_error(trial, normalized_error)
        )
        and (
            normalized_error_contains is None
            or _trial_matches_error_contains(trial, normalized_error_contains)
        )
        and (
            normalized_error_type is None
            or _trial_matches_error_type(trial, normalized_error_type)
        )
    ]
    for trial in selected_trials:
        trial["duration_seconds"] = _trial_duration_seconds(trial)
    if frontier_only:
        frontier_trial_indices = {
            int(entry["trial_index"])
            for entry in _selected_objective_duration_frontier(
                selected_trials,
                objective_mode=objective_mode,
            )
        }
        selected_trials = [
            trial
            for trial in selected_trials
            if _trial_index_key(trial) in frontier_trial_indices
        ]

    if sort_by == "trial-index":
        selected_trials.sort(key=_trial_index_key, reverse=descending)
    elif sort_by == "objective-value":
        with_objective = [trial for trial in selected_trials if trial.get("objective_value") is not None]
        without_objective = [trial for trial in selected_trials if trial.get("objective_value") is None]
        with_objective.sort(
            key=lambda trial: (float(trial["objective_value"]), _trial_index_key(trial)),
            reverse=descending,
        )
        without_objective.sort(key=_trial_index_key)
        selected_trials = with_objective + without_objective
    else:
        with_duration = [trial for trial in selected_trials if trial.get("duration_seconds") is not None]
        without_duration = [trial for trial in selected_trials if trial.get("duration_seconds") is None]
        if descending:
            with_duration.sort(
                key=lambda trial: (-float(trial["duration_seconds"]), _trial_index_key(trial))
            )
        else:
            with_duration.sort(
                key=lambda trial: (float(trial["duration_seconds"]), _trial_index_key(trial))
            )
        without_duration.sort(key=_trial_index_key)
        selected_trials = with_duration + without_duration

    if top_k is not None:
        selected_trials = selected_trials[:top_k]

    selected_payload = dict(payload)
    search_space_specs = _study_report_search_space_specs(payload)
    selected_best_trial, selected_best_objective_value = _best_record_from_records(selected_trials, mode=objective_mode)
    selected_best_trial_index = None
    if selected_best_trial is not None:
        selected_best_trial_index = int(selected_best_trial["trial_index"])
    for trial in selected_trials:
        raw_objective_value = trial.get("objective_value")
        objective_value = None if raw_objective_value is None else float(raw_objective_value)
        trial["selected_best_objective_delta"] = _objective_delta(
            objective_value,
            best_value=selected_best_objective_value,
            mode=objective_mode,
        )
    selected_incumbent_trace = _selected_incumbent_trace(
        selected_trials,
        objective_mode=objective_mode,
    )
    selected_incumbent_trace_by_trial_index = {
        int(entry["trial_index"]): entry for entry in selected_incumbent_trace
    }
    for trial in selected_trials:
        trace_entry = selected_incumbent_trace_by_trial_index.get(_trial_index_key(trial), {})
        trial["selected_incumbent_trial_index"] = trace_entry.get("selected_incumbent_trial_index")
        trial["selected_incumbent_objective_value"] = trace_entry.get("selected_incumbent_objective_value")
        trial["selected_is_incumbent_update"] = trace_entry.get("selected_is_incumbent_update", False)
        trial["selected_incumbent_update_improvement"] = trace_entry.get("selected_incumbent_update_improvement")
        trial["selected_incumbent_trials_since_previous_update"] = trace_entry.get(
            "selected_incumbent_trials_since_previous_update"
        )
        trial["selected_incumbent_age_trials"] = trace_entry.get("selected_incumbent_age_trials")
        trial["selected_incumbent_age_seconds"] = trace_entry.get("selected_incumbent_age_seconds")
    selected_objective_duration_frontier = _selected_objective_duration_frontier(
        selected_trials,
        objective_mode=objective_mode,
    )
    selected_objective_duration_frontier_indices = {
        int(entry["trial_index"]) for entry in selected_objective_duration_frontier
    }
    for trial in selected_trials:
        trial["is_objective_duration_frontier"] = (
            _trial_index_key(trial) in selected_objective_duration_frontier_indices
        )
    selected_payload["trials"] = selected_trials
    selected_payload["selected_trial_count"] = len(selected_trials)
    selected_payload["selected_best_trial_index"] = selected_best_trial_index
    selected_payload["selected_best_objective_value"] = selected_best_objective_value
    selected_payload["selected_status_counts"] = _status_counts_for_trials(selected_trials)
    selected_payload["selected_objective_summary"] = _selected_objective_summary(selected_trials)
    selected_payload["selected_duration_summary"] = _selected_duration_summary(selected_trials)
    selected_payload["selected_incumbent_trace"] = selected_incumbent_trace
    selected_payload["selected_incumbent_update_summary"] = _selected_incumbent_update_summary(
        selected_incumbent_trace
    )
    selected_payload["selected_incumbent_staleness_summary"] = _selected_incumbent_staleness_summary(
        selected_incumbent_trace
    )
    selected_payload["selected_objective_duration_frontier"] = selected_objective_duration_frontier
    selected_payload["selected_error_summaries"] = _selected_error_summaries(selected_trials)
    selected_payload["selected_error_type_summaries"] = _selected_error_type_summaries(selected_trials)
    selected_payload["selected_parameter_summaries"] = _selected_parameter_summaries(
        selected_trials,
        objective_mode=objective_mode,
        search_space_specs=search_space_specs,
    )
    selected_payload["selected_parameter_value_summaries"] = _selected_parameter_value_summaries(
        selected_trials,
        objective_mode=objective_mode,
        selected_best_objective_value=selected_best_objective_value,
    )
    selected_payload["selected_parameter_incumbent_summaries"] = _selected_parameter_incumbent_summaries(
        selected_payload["selected_parameter_value_summaries"]
    )
    selected_payload["selected_parameter_incumbent_leaderboard"] = _selected_parameter_incumbent_leaderboard(
        selected_payload["selected_parameter_incumbent_summaries"]
    )
    selected_payload["selected_parameter_effect_leaderboard"] = _selected_parameter_effect_leaderboard(
        selected_payload["selected_parameter_value_summaries"],
        objective_mode=objective_mode,
    )
    if focus_param is not None:
        normalized_focus_param = str(focus_param)
        normalized_focus_sort_by = "best-objective-value" if focus_sort_by is None else str(focus_sort_by)
        selected_payload["focused_parameter_name"] = normalized_focus_param
        selected_payload["focused_parameter_value_summary"] = _focused_parameter_value_summary(
            selected_payload["selected_parameter_value_summaries"],
            focus_param=normalized_focus_param,
            focus_sort_by=normalized_focus_sort_by,
            focus_top_k=focus_top_k,
        )
    selected_payload["search_efficiency_summary"] = _search_efficiency_summary(
        trials=selected_trials,
        selected_trial_count=len(selected_trials),
        selected_best_trial_index=selected_best_trial_index,
        selected_best_objective_value=selected_best_objective_value,
        selected_objective_summary=selected_payload["selected_objective_summary"],
        selected_parameter_summaries=selected_payload["selected_parameter_summaries"],
        objective_mode=objective_mode,
    )
    report_filters: dict[str, object] = {
        "status": status,
        "sort_by": sort_by,
        "descending": descending,
        "top_k": top_k,
    }
    if frontier_only:
        report_filters["frontier_only"] = True
    if normalized_objective_at_least is not None:
        report_filters["objective_at_least"] = normalized_objective_at_least
    if normalized_objective_at_most is not None:
        report_filters["objective_at_most"] = normalized_objective_at_most
    if normalized_duration_at_least is not None:
        report_filters["duration_at_least"] = normalized_duration_at_least
    if normalized_duration_at_most is not None:
        report_filters["duration_at_most"] = normalized_duration_at_most
    if normalized_param_filters is not None:
        report_filters["param_filters"] = dict(normalized_param_filters)
    if normalized_error is not None:
        report_filters["error"] = normalized_error
    if normalized_error_contains is not None:
        report_filters["error_contains"] = normalized_error_contains
    if normalized_error_type is not None:
        report_filters["error_type"] = normalized_error_type
    if focus_param is not None:
        report_filters["focus_param"] = str(focus_param)
        report_filters["focus_sort_by"] = "best-objective-value" if focus_sort_by is None else str(focus_sort_by)
        if focus_top_k is not None:
            report_filters["focus_top_k"] = int(focus_top_k)
    selected_payload["report_filters"] = report_filters
    return selected_payload


def render_text_study_report(payload: Mapping[str, Any]) -> str:
    objective = payload.get("objective", {})
    if not isinstance(objective, Mapping):
        objective = {}
    search_efficiency_summary = payload.get("search_efficiency_summary", {})
    if not isinstance(search_efficiency_summary, Mapping):
        search_efficiency_summary = {}
    report_filters = payload.get("report_filters")
    lines = [
        f"study_name={payload.get('study_name')}",
        f"study_dir={payload.get('study_dir')}",
        f"backend={payload.get('backend')}",
        f"sampler={payload.get('sampler')}",
        f"objective_metric={objective.get('metric')}",
        f"objective_mode={objective.get('mode')}",
        f"trial_count={payload.get('trial_count')}",
        f"selected_trial_count={payload.get('selected_trial_count', payload.get('trial_count'))}",
        "status_counts="
        f"{json.dumps(payload.get('status_counts', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_status_counts="
        f"{json.dumps(payload.get('selected_status_counts', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_objective_summary="
        f"{json.dumps(payload.get('selected_objective_summary', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_duration_summary="
        f"{json.dumps(payload.get('selected_duration_summary', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_incumbent_trace="
        f"{json.dumps(payload.get('selected_incumbent_trace', []), sort_keys=True, ensure_ascii=False)}",
        "selected_incumbent_update_summary="
        f"{json.dumps(payload.get('selected_incumbent_update_summary', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_incumbent_staleness_summary="
        f"{json.dumps(payload.get('selected_incumbent_staleness_summary', {}), sort_keys=True, ensure_ascii=False)}",
        "selected_parameter_incumbent_leaderboard="
        f"{json.dumps(payload.get('selected_parameter_incumbent_leaderboard', []), sort_keys=True, ensure_ascii=False)}",
        "selected_parameter_effect_leaderboard="
        f"{json.dumps(payload.get('selected_parameter_effect_leaderboard', []), sort_keys=True, ensure_ascii=False)}",
        "selected_objective_duration_frontier="
        f"{json.dumps(payload.get('selected_objective_duration_frontier', []), sort_keys=True, ensure_ascii=False)}",
        "selected_error_summaries="
        f"{json.dumps(payload.get('selected_error_summaries', []), sort_keys=True, ensure_ascii=False)}",
        "selected_error_type_summaries="
        f"{json.dumps(payload.get('selected_error_type_summaries', []), sort_keys=True, ensure_ascii=False)}",
        "search_efficiency_summary="
        f"{json.dumps(payload.get('search_efficiency_summary', {}), sort_keys=True, ensure_ascii=False)}",
        "search_efficiency_selected_trials_until_best="
        f"{search_efficiency_summary.get('selected_trials_until_best')}",
        "search_efficiency_selected_trial_share_until_best="
        f"{search_efficiency_summary.get('selected_trial_share_until_best')}",
        "search_efficiency_completed_trials_until_best="
        f"{search_efficiency_summary.get('completed_trials_until_best')}",
        "search_efficiency_completed_trial_share_until_best="
        f"{search_efficiency_summary.get('completed_trial_share_until_best')}",
        "search_efficiency_time_to_best_seconds="
        f"{search_efficiency_summary.get('time_to_best_seconds')}",
        f"selected_best_trial_index={payload.get('selected_best_trial_index')}",
        f"selected_best_objective_value={payload.get('selected_best_objective_value')}",
        f"best_trial_index={payload.get('best_trial_index')}",
        f"best_objective_value={payload.get('best_objective_value')}",
        f"best_run_dir={payload.get('best_run_dir')}",
        f"best_checkpoint_path={payload.get('best_checkpoint_path')}",
    ]
    if isinstance(payload.get("config_export_summary"), Mapping):
        lines.append(
            "config_export_summary="
            f"{json.dumps(payload.get('config_export_summary', {}), sort_keys=True, ensure_ascii=False)}"
        )
    if isinstance(report_filters, Mapping):
        lines.append(f"report_filters={json.dumps(dict(report_filters), sort_keys=True, ensure_ascii=False)}")
    selected_parameter_summaries = payload.get("selected_parameter_summaries", {})
    if isinstance(selected_parameter_summaries, Mapping):
        for param_name in sorted(str(key) for key in selected_parameter_summaries):
            lines.append(
                f"selected_parameter_summary[{param_name}]="
                f"{json.dumps(selected_parameter_summaries[param_name], sort_keys=True, ensure_ascii=False)}"
            )
    selected_parameter_incumbent_summaries = payload.get("selected_parameter_incumbent_summaries", {})
    if isinstance(selected_parameter_incumbent_summaries, Mapping):
        for param_name in sorted(str(key) for key in selected_parameter_incumbent_summaries):
            lines.append(
                f"selected_parameter_incumbent_summary[{param_name}]="
                f"{json.dumps(selected_parameter_incumbent_summaries[param_name], sort_keys=True, ensure_ascii=False)}"
            )
    selected_parameter_incumbent_leaderboard = payload.get("selected_parameter_incumbent_leaderboard", [])
    if isinstance(selected_parameter_incumbent_leaderboard, list) and selected_parameter_incumbent_leaderboard:
        lines.append("")
        lines.append("[parameter incumbent leaderboard]")
        for entry in selected_parameter_incumbent_leaderboard:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[parameter incumbent {entry.get('name')}]")
            lines.append(f"name={entry.get('name')}")
            lines.append(f"incumbent_update_count={entry.get('incumbent_update_count')}")
            lines.append(f"contributing_values={json.dumps(entry.get('contributing_values', []), ensure_ascii=False)}")
            lines.append(f"contributing_value_count={entry.get('contributing_value_count')}")
            lines.append(f"top_incumbent_value={entry.get('top_incumbent_value')}")
            lines.append(f"top_incumbent_value_updates={entry.get('top_incumbent_value_updates')}")
            lines.append(f"latest_incumbent_value={entry.get('latest_incumbent_value')}")
            lines.append(f"latest_incumbent_trial_index={entry.get('latest_incumbent_trial_index')}")
    selected_parameter_effect_leaderboard = payload.get("selected_parameter_effect_leaderboard", [])
    if isinstance(selected_parameter_effect_leaderboard, list) and selected_parameter_effect_leaderboard:
        lines.append("")
        lines.append("[parameter effect leaderboard]")
        for entry in selected_parameter_effect_leaderboard:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[parameter effect {entry.get('name')}]")
            lines.append(f"name={entry.get('name')}")
            lines.append(f"observed_value_count={entry.get('observed_value_count')}")
            lines.append(f"completed_value_count={entry.get('completed_value_count')}")
            lines.append(f"best_objective_spread={entry.get('best_objective_spread')}")
            lines.append(f"mean_objective_spread={entry.get('mean_objective_spread')}")
            lines.append(f"top_value_by_best_objective={entry.get('top_value_by_best_objective')}")
            lines.append(f"top_best_objective_value={entry.get('top_best_objective_value')}")
            lines.append(f"bottom_value_by_best_objective={entry.get('bottom_value_by_best_objective')}")
            lines.append(f"bottom_best_objective_value={entry.get('bottom_best_objective_value')}")
            lines.append(f"top_value_by_mean_objective={entry.get('top_value_by_mean_objective')}")
            lines.append(f"top_mean_objective_value={entry.get('top_mean_objective_value')}")
            lines.append(f"bottom_value_by_mean_objective={entry.get('bottom_value_by_mean_objective')}")
            lines.append(f"bottom_mean_objective_value={entry.get('bottom_mean_objective_value')}")
    selected_parameter_value_summaries = payload.get("selected_parameter_value_summaries", {})
    if isinstance(selected_parameter_value_summaries, Mapping):
        for param_name in sorted(str(key) for key in selected_parameter_value_summaries):
            lines.append(
                f"selected_parameter_value_summary[{param_name}]="
                f"{json.dumps(selected_parameter_value_summaries[param_name], sort_keys=True, ensure_ascii=False)}"
            )
    selected_error_summaries = payload.get("selected_error_summaries", [])
    if isinstance(selected_error_summaries, list) and selected_error_summaries:
        lines.append("")
        lines.append("[failed trial errors]")
        for entry in selected_error_summaries:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[error {entry.get('error')}]")
            lines.append(f"error={entry.get('error')}")
            lines.append(f"failed_trials={entry.get('failed_trials')}")
            lines.append(f"selected_trial_share={entry.get('selected_trial_share')}")
            lines.append(f"failed_trial_share={entry.get('failed_trial_share')}")
            lines.append(f"trial_indices={json.dumps(entry.get('trial_indices', []), ensure_ascii=False)}")
    selected_error_type_summaries = payload.get("selected_error_type_summaries", [])
    if isinstance(selected_error_type_summaries, list) and selected_error_type_summaries:
        lines.append("")
        lines.append("[failed trial error types]")
        for entry in selected_error_type_summaries:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[error type {entry.get('error_type')}]")
            lines.append(f"error_type={entry.get('error_type')}")
            lines.append(f"errors={json.dumps(entry.get('errors', []), ensure_ascii=False)}")
            lines.append(f"failed_trials={entry.get('failed_trials')}")
            lines.append(f"selected_trial_share={entry.get('selected_trial_share')}")
            lines.append(f"failed_trial_share={entry.get('failed_trial_share')}")
            lines.append(f"trial_indices={json.dumps(entry.get('trial_indices', []), ensure_ascii=False)}")
    if payload.get("focused_parameter_name") is not None:
        lines.append(f"focused_parameter_name={payload.get('focused_parameter_name')}")
        lines.append(
            "focused_parameter_value_summary="
            f"{json.dumps(payload.get('focused_parameter_value_summary', []), sort_keys=True, ensure_ascii=False)}"
        )
        focused_entries = payload.get("focused_parameter_value_summary", [])
        if isinstance(focused_entries, list):
            lines.append("")
            lines.append(f"[focused parameter {payload.get('focused_parameter_name')}]")
            for entry in focused_entries:
                if not isinstance(entry, Mapping):
                    continue
                lines.append("")
                lines.append(f"[focused value {entry.get('value')}]")
                lines.append(f"value={entry.get('value')}")
                lines.append(f"trial_count={entry.get('trial_count')}")
                lines.append(f"completed_trials={entry.get('completed_trials')}")
                lines.append(f"failed_trials={entry.get('failed_trials')}")
                lines.append(f"timed_trials={entry.get('timed_trials')}")
                lines.append(f"untimed_trials={entry.get('untimed_trials')}")
                lines.append(f"completion_rate={entry.get('completion_rate')}")
                lines.append(f"failure_rate={entry.get('failure_rate')}")
                lines.append(f"best_objective_value={entry.get('best_objective_value')}")
                lines.append(f"mean_objective_value={entry.get('mean_objective_value')}")
                lines.append(f"median_objective_value={entry.get('median_objective_value')}")
                lines.append(f"min_duration_seconds={entry.get('min_duration_seconds')}")
                lines.append(f"max_duration_seconds={entry.get('max_duration_seconds')}")
                lines.append(f"mean_duration_seconds={entry.get('mean_duration_seconds')}")
                lines.append(f"median_duration_seconds={entry.get('median_duration_seconds')}")
                lines.append(f"incumbent_updates={entry.get('incumbent_updates')}")
                lines.append(f"latest_incumbent_trial_index={entry.get('latest_incumbent_trial_index')}")
                lines.append(f"selected_best_objective_delta={entry.get('selected_best_objective_delta')}")
                lines.append(f"rank_by_best_objective_value={entry.get('rank_by_best_objective_value')}")
                lines.append(f"rank_by_mean_objective_value={entry.get('rank_by_mean_objective_value')}")
    selected_incumbent_trace = payload.get("selected_incumbent_trace", [])
    if isinstance(selected_incumbent_trace, list) and selected_incumbent_trace:
        lines.append("")
        lines.append("[incumbent trace]")
        for entry in selected_incumbent_trace:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[incumbent step {entry.get('trial_index')}]")
            lines.append(f"trial_index={entry.get('trial_index')}")
            lines.append(f"status={entry.get('status')}")
            lines.append(f"objective_value={entry.get('objective_value')}")
            lines.append(f"selected_incumbent_trial_index={entry.get('selected_incumbent_trial_index')}")
            lines.append(f"selected_incumbent_objective_value={entry.get('selected_incumbent_objective_value')}")
            lines.append(f"selected_is_incumbent_update={entry.get('selected_is_incumbent_update')}")
            lines.append(f"selected_incumbent_update_improvement={entry.get('selected_incumbent_update_improvement')}")
            lines.append(
                "selected_incumbent_trials_since_previous_update="
                f"{entry.get('selected_incumbent_trials_since_previous_update')}"
            )
            lines.append(f"selected_incumbent_age_trials={entry.get('selected_incumbent_age_trials')}")
            lines.append(f"selected_incumbent_age_seconds={entry.get('selected_incumbent_age_seconds')}")
    selected_objective_duration_frontier = payload.get("selected_objective_duration_frontier", [])
    if isinstance(selected_objective_duration_frontier, list) and selected_objective_duration_frontier:
        lines.append("")
        lines.append("[objective-duration frontier]")
        for entry in selected_objective_duration_frontier:
            if not isinstance(entry, Mapping):
                continue
            lines.append("")
            lines.append(f"[frontier trial {entry.get('trial_index')}]")
            lines.append(f"trial_index={entry.get('trial_index')}")
            lines.append(f"objective_value={entry.get('objective_value')}")
            lines.append(f"duration_seconds={entry.get('duration_seconds')}")
            lines.append(f"selected_best_objective_delta={entry.get('selected_best_objective_delta')}")
            lines.append(
                "params="
                f"{json.dumps(entry.get('params', {}), sort_keys=True, ensure_ascii=False)}"
            )
    trials = payload.get("trials", [])
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            lines.append("")
            lines.append(f"[trial {trial.get('trial_index')}]")
            lines.append(f"status={trial.get('status')}")
            lines.append(f"objective_value={trial.get('objective_value')}")
            lines.append(f"duration_seconds={trial.get('duration_seconds')}")
            lines.append(f"selected_best_objective_delta={trial.get('selected_best_objective_delta')}")
            lines.append(f"run_dir={trial.get('run_dir')}")
            lines.append(f"checkpoint_path={trial.get('checkpoint_path')}")
            lines.append(
                "params="
                f"{json.dumps(trial.get('params', {}), sort_keys=True, ensure_ascii=False)}"
            )
            if trial.get("error") is not None:
                lines.append(f"error={trial.get('error')}")
    return "\n".join(lines) + "\n"


def render_json_study_report(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _study_report_param_keys(payload: Mapping[str, Any]) -> list[str]:
    trials = payload.get("trials", [])
    if not isinstance(trials, list):
        return []
    keys: set[str] = set()
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        params = trial.get("params", {})
        if not isinstance(params, Mapping):
            continue
        keys.update(str(key) for key in params)
    return sorted(keys)


def csv_study_report_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    objective = payload.get("objective", {})
    if not isinstance(objective, Mapping):
        objective = {}
    search_efficiency_summary = payload.get("search_efficiency_summary", {})
    if not isinstance(search_efficiency_summary, Mapping):
        search_efficiency_summary = {}
    param_keys = _study_report_param_keys(payload)
    focused_parameter_name = payload.get("focused_parameter_name")
    focused_parameter_value_summary = payload.get("focused_parameter_value_summary", [])
    focused_summary_by_value: dict[str, Mapping[str, object]] = {}
    if isinstance(focused_parameter_name, str) and isinstance(focused_parameter_value_summary, list):
        for entry in focused_parameter_value_summary:
            if not isinstance(entry, Mapping):
                continue
            value_key = json.dumps(entry.get("value"), sort_keys=True, ensure_ascii=False)
            focused_summary_by_value[value_key] = entry
    common = {
        "study_name": payload.get("study_name"),
        "study_dir": payload.get("study_dir"),
        "backend": payload.get("backend"),
        "sampler": payload.get("sampler"),
        "objective_metric": objective.get("metric"),
        "objective_mode": objective.get("mode"),
        "base_config_path": payload.get("base_config_path"),
        "output_dir": payload.get("output_dir"),
        "trial_count": payload.get("trial_count"),
        "selected_trial_count": payload.get("selected_trial_count", payload.get("trial_count")),
        "selected_best_trial_index": payload.get("selected_best_trial_index"),
        "selected_best_objective_value": payload.get("selected_best_objective_value"),
        "best_trial_index": payload.get("best_trial_index"),
        "best_objective_value": payload.get("best_objective_value"),
        "best_run_dir": payload.get("best_run_dir"),
        "best_checkpoint_path": payload.get("best_checkpoint_path"),
        "status_counts_json": json.dumps(payload.get("status_counts", {}), sort_keys=True, ensure_ascii=False),
        "selected_status_counts_json": json.dumps(
            payload.get("selected_status_counts", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_incumbent_trace_json": json.dumps(
            payload.get("selected_incumbent_trace", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_incumbent_update_summary_json": json.dumps(
            payload.get("selected_incumbent_update_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_incumbent_staleness_summary_json": json.dumps(
            payload.get("selected_incumbent_staleness_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_objective_duration_frontier_json": json.dumps(
            payload.get("selected_objective_duration_frontier", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_duration_summary_json": json.dumps(
            payload.get("selected_duration_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_error_summaries_json": json.dumps(
            payload.get("selected_error_summaries", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_error_type_summaries_json": json.dumps(
            payload.get("selected_error_type_summaries", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_objective_summary_json": json.dumps(
            payload.get("selected_objective_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_parameter_incumbent_leaderboard_json": json.dumps(
            payload.get("selected_parameter_incumbent_leaderboard", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_parameter_effect_leaderboard_json": json.dumps(
            payload.get("selected_parameter_effect_leaderboard", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_parameter_incumbent_summaries_json": json.dumps(
            payload.get("selected_parameter_incumbent_summaries", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "selected_parameter_value_summaries_json": json.dumps(
            payload.get("selected_parameter_value_summaries", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "focused_parameter_name": payload.get("focused_parameter_name"),
        "focused_parameter_value_summary_json": json.dumps(
            payload.get("focused_parameter_value_summary", []),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "search_efficiency_summary_json": json.dumps(
            payload.get("search_efficiency_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "search_efficiency_selected_trials_until_best": search_efficiency_summary.get("selected_trials_until_best"),
        "search_efficiency_selected_trial_share_until_best": search_efficiency_summary.get(
            "selected_trial_share_until_best"
        ),
        "search_efficiency_completed_trials_until_best": search_efficiency_summary.get(
            "completed_trials_until_best"
        ),
        "search_efficiency_completed_trial_share_until_best": search_efficiency_summary.get(
            "completed_trial_share_until_best"
        ),
        "search_efficiency_time_to_best_seconds": search_efficiency_summary.get("time_to_best_seconds"),
        "config_export_summary_json": json.dumps(
            payload.get("config_export_summary", {}),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "report_filters_json": json.dumps(payload.get("report_filters", {}), sort_keys=True, ensure_ascii=False),
    }
    rows: list[dict[str, object]] = []
    trials = payload.get("trials", [])
    if not isinstance(trials, list):
        return rows
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        params = trial.get("params", {})
        params_mapping = params if isinstance(params, Mapping) else {}
        row: dict[str, object] = dict(common)
        row.update(
            {
                "trial_index": trial.get("trial_index"),
                "status": trial.get("status"),
                "objective_value": trial.get("objective_value"),
                "duration_seconds": trial.get("duration_seconds"),
                "selected_incumbent_trial_index": trial.get("selected_incumbent_trial_index"),
                "selected_incumbent_objective_value": trial.get("selected_incumbent_objective_value"),
                "selected_is_incumbent_update": trial.get("selected_is_incumbent_update", False),
                "selected_incumbent_update_improvement": trial.get("selected_incumbent_update_improvement"),
                "selected_incumbent_trials_since_previous_update": trial.get(
                    "selected_incumbent_trials_since_previous_update"
                ),
                "selected_incumbent_age_trials": trial.get("selected_incumbent_age_trials"),
                "selected_incumbent_age_seconds": trial.get("selected_incumbent_age_seconds"),
                "is_objective_duration_frontier": trial.get("is_objective_duration_frontier", False),
                "selected_best_objective_delta": trial.get("selected_best_objective_delta"),
                "run_dir": trial.get("run_dir"),
                "checkpoint_path": trial.get("checkpoint_path"),
                "error": trial.get("error"),
                "started_at": trial.get("started_at"),
                "ended_at": trial.get("ended_at"),
                "params_json": json.dumps(params_mapping, sort_keys=True, ensure_ascii=False),
            }
        )
        focused_entry: Mapping[str, object] | None = None
        if isinstance(focused_parameter_name, str) and focused_parameter_name in params_mapping:
            focused_entry = focused_summary_by_value.get(
                json.dumps(params_mapping.get(focused_parameter_name), sort_keys=True, ensure_ascii=False)
            )
        row.update(
            {
                "focused_parameter_value": None if focused_entry is None else focused_entry.get("value"),
                "focused_parameter_trial_count": None if focused_entry is None else focused_entry.get("trial_count"),
                "focused_parameter_completed_trials": (
                    None if focused_entry is None else focused_entry.get("completed_trials")
                ),
                "focused_parameter_failed_trials": (
                    None if focused_entry is None else focused_entry.get("failed_trials")
                ),
                "focused_parameter_timed_trials": (
                    None if focused_entry is None else focused_entry.get("timed_trials")
                ),
                "focused_parameter_untimed_trials": (
                    None if focused_entry is None else focused_entry.get("untimed_trials")
                ),
                "focused_parameter_completion_rate": (
                    None if focused_entry is None else focused_entry.get("completion_rate")
                ),
                "focused_parameter_failure_rate": (
                    None if focused_entry is None else focused_entry.get("failure_rate")
                ),
                "focused_parameter_best_objective_value": (
                    None if focused_entry is None else focused_entry.get("best_objective_value")
                ),
                "focused_parameter_mean_objective_value": (
                    None if focused_entry is None else focused_entry.get("mean_objective_value")
                ),
                "focused_parameter_median_objective_value": (
                    None if focused_entry is None else focused_entry.get("median_objective_value")
                ),
                "focused_parameter_min_duration_seconds": (
                    None if focused_entry is None else focused_entry.get("min_duration_seconds")
                ),
                "focused_parameter_max_duration_seconds": (
                    None if focused_entry is None else focused_entry.get("max_duration_seconds")
                ),
                "focused_parameter_mean_duration_seconds": (
                    None if focused_entry is None else focused_entry.get("mean_duration_seconds")
                ),
                "focused_parameter_median_duration_seconds": (
                    None if focused_entry is None else focused_entry.get("median_duration_seconds")
                ),
                "focused_parameter_incumbent_updates": (
                    None if focused_entry is None else focused_entry.get("incumbent_updates")
                ),
                "focused_parameter_latest_incumbent_trial_index": (
                    None if focused_entry is None else focused_entry.get("latest_incumbent_trial_index")
                ),
                "focused_parameter_selected_best_objective_delta": (
                    None if focused_entry is None else focused_entry.get("selected_best_objective_delta")
                ),
                "focused_parameter_rank_by_best_objective_value": (
                    None if focused_entry is None else focused_entry.get("rank_by_best_objective_value")
                ),
                "focused_parameter_rank_by_mean_objective_value": (
                    None if focused_entry is None else focused_entry.get("rank_by_mean_objective_value")
                ),
            }
        )
        for key in param_keys:
            row[f"param_{key}"] = params_mapping.get(key)
        rows.append(row)
    return rows


def render_csv_study_report(payload: Mapping[str, Any]) -> str:
    rows = csv_study_report_rows(payload)
    fieldnames = [
        "study_name",
        "backend",
        "sampler",
        "objective_metric",
        "objective_mode",
        "study_dir",
        "base_config_path",
        "output_dir",
        "trial_count",
        "selected_trial_count",
        "selected_best_trial_index",
        "selected_best_objective_value",
        "best_trial_index",
        "best_objective_value",
        "best_run_dir",
        "best_checkpoint_path",
        "status_counts_json",
        "selected_status_counts_json",
        "selected_incumbent_trace_json",
        "selected_incumbent_update_summary_json",
        "selected_incumbent_staleness_summary_json",
        "selected_objective_duration_frontier_json",
        "selected_duration_summary_json",
        "selected_error_summaries_json",
        "selected_error_type_summaries_json",
        "selected_objective_summary_json",
        "selected_parameter_incumbent_leaderboard_json",
        "selected_parameter_effect_leaderboard_json",
        "selected_parameter_incumbent_summaries_json",
        "selected_parameter_value_summaries_json",
        "focused_parameter_name",
        "focused_parameter_value_summary_json",
        "focused_parameter_value",
        "focused_parameter_trial_count",
        "focused_parameter_completed_trials",
        "focused_parameter_failed_trials",
        "focused_parameter_timed_trials",
        "focused_parameter_untimed_trials",
        "focused_parameter_completion_rate",
        "focused_parameter_failure_rate",
        "focused_parameter_best_objective_value",
        "focused_parameter_mean_objective_value",
        "focused_parameter_median_objective_value",
        "focused_parameter_min_duration_seconds",
        "focused_parameter_max_duration_seconds",
        "focused_parameter_mean_duration_seconds",
        "focused_parameter_median_duration_seconds",
        "focused_parameter_incumbent_updates",
        "focused_parameter_latest_incumbent_trial_index",
        "focused_parameter_selected_best_objective_delta",
        "focused_parameter_rank_by_best_objective_value",
        "focused_parameter_rank_by_mean_objective_value",
        "search_efficiency_summary_json",
        "search_efficiency_selected_trials_until_best",
        "search_efficiency_selected_trial_share_until_best",
        "search_efficiency_completed_trials_until_best",
        "search_efficiency_completed_trial_share_until_best",
        "search_efficiency_time_to_best_seconds",
        "config_export_summary_json",
        "report_filters_json",
        "trial_index",
        "status",
        "objective_value",
        "duration_seconds",
        "selected_incumbent_trial_index",
        "selected_incumbent_objective_value",
        "selected_is_incumbent_update",
        "selected_incumbent_update_improvement",
        "selected_incumbent_trials_since_previous_update",
        "selected_incumbent_age_trials",
        "selected_incumbent_age_seconds",
        "is_objective_duration_frontier",
        "selected_best_objective_delta",
        "run_dir",
        "checkpoint_path",
        "error",
        "started_at",
        "ended_at",
        "params_json",
        *[f"param_{key}" for key in _study_report_param_keys(payload)],
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _best_record_from_records(records: list[dict[str, Any]], *, mode: str) -> tuple[dict[str, Any] | None, float | None]:
    best_record: dict[str, Any] | None = None
    best_value: float | None = None
    for record in records:
        if record.get("status") != "completed":
            continue
        objective_value = record.get("objective_value")
        if objective_value is None:
            continue
        candidate = float(objective_value)
        if _is_better(candidate, best_value, mode=mode):
            best_value = candidate
            best_record = record
    return best_record, best_value


def _best_config_payload_from_record(best_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if best_record is None:
        return None
    run_dir = best_record.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {config_path}, got {type(payload)!r}")
    return payload


def _write_study_outputs(
    *,
    config: StudyConfig,
    study_dir: Path,
    trial_records: list[dict[str, Any]],
    best_record: dict[str, Any] | None,
    best_config_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    study_json_path = study_dir / "study.json"
    best_trial_path = study_dir / "best_trial.json"
    best_config_path = study_dir / "best_config.yaml"
    status_counts = _status_counts(trial_records)
    best_objective_value = None if best_record is None else float(best_record["objective_value"])
    study_payload = {
        "study_name": config.study.name,
        "backend": config.study.backend,
        "sampler": config.study.sampler,
        "objective": {
            "metric": config.study.objective.metric,
            "mode": config.study.objective.mode,
        },
        "base_config_path": str(config.base_config_path),
        "output_dir": str(config.output_dir),
        "trial_count": len(trial_records),
        "status_counts": status_counts,
        "best_trial_index": None if best_record is None else int(best_record["trial_index"]),
        "best_objective_value": best_objective_value,
        "best_run_dir": None if best_record is None else best_record.get("run_dir"),
        "best_checkpoint_path": None if best_record is None else best_record.get("checkpoint_path"),
        "study_config": serialize_study_config(config),
    }
    _write_json(study_json_path, study_payload)

    if best_record is not None:
        _write_json(best_trial_path, best_record)
        resolved_best_config = best_config_payload or _best_config_payload_from_record(best_record)
        if resolved_best_config is not None:
            best_config_path.write_text(yaml.safe_dump(resolved_best_config, sort_keys=False), encoding="utf-8")
    return study_payload


def _run_native_study(config: StudyConfig, *, study_dir: Path, existing_records: list[dict[str, Any]] | None = None) -> StudyResult:
    trials_jsonl_path = study_dir / "trials.jsonl"
    trials_dir = study_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_records = list(existing_records or [])
    trial_param_sets = _iter_trial_params(config)
    existing_by_index = {int(record["trial_index"]): record for record in trial_records}
    best_record, best_objective_value = _best_record_from_records(trial_records, mode=config.study.objective.mode)
    best_config_payload = _best_config_payload_from_record(best_record)
    manager = DefaultExperimentManager()

    for trial_index, params in enumerate(trial_param_sets):
        if trial_index in existing_by_index:
            continue
        started_at = _utc_now()
        trial_config = _apply_trial_params(config.base_train_config, params, trial_output_dir=trials_dir)
        record: dict[str, Any] = {
            "trial_index": trial_index,
            "status": "completed",
            "params": params,
            "objective_value": None,
            "run_dir": None,
            "checkpoint_path": None,
            "error": None,
            "started_at": started_at,
            "ended_at": None,
        }
        try:
            result = manager.setup(trial_config).train()
            record["run_dir"] = str(result.run_dir)
            record["checkpoint_path"] = None if result.checkpoint_path is None else str(result.checkpoint_path)
            metric_value = result.metrics.get(config.study.objective.metric)
            if metric_value is None:
                raise ValueError(
                    f"trial {trial_index} did not produce objective metric '{config.study.objective.metric}'"
                )
            objective_value = float(metric_value)
            record["objective_value"] = objective_value
            if _is_better(objective_value, best_objective_value, mode=config.study.objective.mode):
                best_objective_value = objective_value
                best_record = dict(record)
                best_config_payload = serialize_train_config(trial_config)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if config.study.fail_fast:
                record["ended_at"] = _utc_now()
                trial_records.append(record)
                _append_trial_record(trials_jsonl_path, record)
                _write_study_outputs(
                    config=config,
                    study_dir=study_dir,
                    trial_records=trial_records,
                    best_record=best_record,
                    best_config_payload=best_config_payload,
                )
                raise
        record["ended_at"] = _utc_now()
        trial_records.append(record)
        _append_trial_record(trials_jsonl_path, record)

    study_payload = _write_study_outputs(
        config=config,
        study_dir=study_dir,
        trial_records=trial_records,
        best_record=best_record,
        best_config_payload=best_config_payload,
    )
    return _study_result_from_payload(study_dir, study_payload)


def run_study(config: StudyConfig) -> StudyResult:
    study_dir = config.output_dir / config.study.name
    study_json_path = study_dir / "study.json"

    if study_json_path.exists():
        raise FileExistsError(f"study already exists at {study_dir}")

    if config.study.backend == "optuna":
        from rl_training.tuning.optuna_backend import run_optuna_study

        return run_optuna_study(config)
    return _run_native_study(config, study_dir=study_dir)


def resume_study(study_dir: str | Path) -> StudyResult:
    resolved_study_dir, payload = _load_study_summary_payload(study_dir)
    study_config_payload = payload.get("study_config")
    if not isinstance(study_config_payload, dict):
        raise ValueError(f"study summary {resolved_study_dir / 'study.json'} is missing a serialized study_config")
    config = deserialize_study_config(study_config_payload)
    trial_records = _load_trial_records(resolved_study_dir / "trials.jsonl")
    if config.study.backend == "optuna":
        from rl_training.tuning.optuna_backend import resume_optuna_study

        return resume_optuna_study(config, study_dir=resolved_study_dir, existing_records=trial_records)
    return _run_native_study(config, study_dir=resolved_study_dir, existing_records=trial_records)
