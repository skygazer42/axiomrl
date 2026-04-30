from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_training.algorithms.base import UpdateResult
from rl_training.cli_config import serialize_train_config
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.runtime.callbacks import Callback
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.trainer import TrainerState, TrainResult
from rl_training.runtime.types import MetricDict
from rl_training.tuning.config import SearchSpaceSpec, StudyConfig
from rl_training.tuning.study import (
    StudyResult,
    _append_trial_record,
    _apply_trial_params,
    _best_config_payload_from_record,
    _best_record_from_records,
    _is_better,
    _study_result_from_payload,
    _utc_now,
    _write_study_outputs,
)


def _import_optuna() -> Any:
    import optuna

    return optuna


@dataclass(slots=True)
class OptunaPruningCallback(Callback):
    trial: object
    metric: str
    prune_exception: Callable[[str], Exception]

    def on_train_start(self, trainer: object) -> None:
        del trainer

    def on_collect_end(self, trainer: object, result: CollectResult) -> None:
        del trainer, result

    def on_update_end(self, trainer: object, result: UpdateResult) -> None:
        del trainer, result

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None:
        if not isinstance(trainer, TrainerState):
            return
        metric_value = metrics.get(self.metric)
        if metric_value is None:
            return
        self.trial.report(float(metric_value), trainer.global_step)
        if self.trial.should_prune():
            trainer.request_stop(
                f"optuna pruning requested on {self.metric} at global_step={trainer.global_step}"
            )
            raise self.prune_exception(
                f"optuna pruning requested on {self.metric} at global_step={trainer.global_step}"
            )

    def on_train_end(self, trainer: object, result: TrainResult) -> None:
        del trainer, result


def _suggest_param(trial: object, path: str, spec: SearchSpaceSpec) -> object:
    if spec.kind == "categorical":
        return trial.suggest_categorical(path, list(spec.values))
    if spec.kind == "int":
        return trial.suggest_int(
            path,
            int(spec.low),
            int(spec.high),
            step=int(1 if spec.step is None else spec.step),
            log=spec.log,
        )
    if spec.kind == "float":
        kwargs: dict[str, object] = {"log": spec.log}
        if spec.step is not None:
            kwargs["step"] = float(spec.step)
        return trial.suggest_float(path, float(spec.low), float(spec.high), **kwargs)
    raise ValueError(f"unsupported search space kind for Optuna: {spec.kind}")


def _distribution_from_spec(optuna: Any, spec: SearchSpaceSpec) -> object:
    if spec.kind == "categorical":
        return optuna.distributions.CategoricalDistribution(list(spec.values))
    if spec.kind == "int":
        return optuna.distributions.IntDistribution(
            low=int(spec.low),
            high=int(spec.high),
            step=int(1 if spec.step is None else spec.step),
            log=spec.log,
        )
    if spec.kind == "float":
        return optuna.distributions.FloatDistribution(
            low=float(spec.low),
            high=float(spec.high),
            step=None if spec.step is None else float(spec.step),
            log=spec.log,
        )
    raise ValueError(f"unsupported search space kind for Optuna distribution: {spec.kind}")


def _replay_existing_trials(optuna: Any, study: object, config: StudyConfig, records: list[dict[str, Any]]) -> None:
    distributions = {
        path: _distribution_from_spec(optuna, spec)
        for path, spec in config.search_space.items()
    }
    for record in records:
        status = str(record.get("status", "failed"))
        params = record.get("params")
        if not isinstance(params, dict):
            params = {}
        state = optuna.trial.TrialState.COMPLETE
        value = None
        if status == "completed":
            value = None if record.get("objective_value") is None else float(record["objective_value"])
        elif status == "pruned":
            state = optuna.trial.TrialState.PRUNED
        else:
            state = optuna.trial.TrialState.FAIL
        frozen_trial = optuna.trial.create_trial(
            state=state,
            value=value,
            params=params,
            distributions=distributions,
        )
        study.add_trial(frozen_trial)


def run_optuna_study(config: StudyConfig) -> StudyResult:
    if config.study.num_trials is None:
        raise ValueError("Optuna studies require study.num_trials")

    study_dir = config.output_dir / config.study.name
    study_json_path = study_dir / "study.json"
    if study_json_path.exists():
        raise FileExistsError(f"study already exists at {study_dir}")
    return resume_optuna_study(config, study_dir=study_dir, existing_records=[])


def resume_optuna_study(config: StudyConfig, *, study_dir: Path, existing_records: list[dict[str, Any]]) -> StudyResult:
    optuna = _import_optuna()
    if config.study.num_trials is None:
        raise ValueError("Optuna studies require study.num_trials")

    trials_dir = study_dir / "trials"
    trials_jsonl_path = study_dir / "trials.jsonl"
    trials_dir.mkdir(parents=True, exist_ok=True)

    direction = "maximize" if config.study.objective.mode == "max" else "minimize"
    if config.study.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=config.study.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=config.study.seed)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=config.study.name)
    _replay_existing_trials(optuna, study, config, existing_records)

    trial_records: list[dict[str, Any]] = list(existing_records)
    existing_by_index = {int(record["trial_index"]): record for record in trial_records}
    best_record, best_objective_value = _best_record_from_records(trial_records, mode=config.study.objective.mode)
    best_config_payload = _best_config_payload_from_record(best_record)
    manager = DefaultExperimentManager()

    for trial_index in range(config.study.num_trials):
        if trial_index in existing_by_index:
            continue
        trial = study.ask()
        params = {path: _suggest_param(trial, path, spec) for path, spec in config.search_space.items()}
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
        pruning_callback = OptunaPruningCallback(
            trial=trial,
            metric=config.study.objective.metric,
            prune_exception=optuna.TrialPruned,
        )
        try:
            result = manager.setup(trial_config, callbacks=[pruning_callback]).train()
            record["run_dir"] = str(result.run_dir)
            record["checkpoint_path"] = None if result.checkpoint_path is None else str(result.checkpoint_path)
            metric_value = result.metrics.get(config.study.objective.metric)
            if metric_value is None:
                raise ValueError(
                    f"trial {trial_index} did not produce objective metric '{config.study.objective.metric}'"
                )
            objective_value = float(metric_value)
            record["objective_value"] = objective_value
            study.tell(trial, objective_value)
            if _is_better(objective_value, best_objective_value, mode=config.study.objective.mode):
                best_objective_value = objective_value
                best_record = dict(record)
                best_config_payload = serialize_train_config(trial_config)
        except optuna.TrialPruned as exc:
            record["status"] = "pruned"
            record["error"] = str(exc)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
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
