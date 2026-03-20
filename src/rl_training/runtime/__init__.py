from __future__ import annotations

from importlib import import_module


_EXPORTS: dict[str, tuple[str, str]] = {
    "CollectResult": ("rl_training.runtime.collector", "CollectResult"),
    "Collector": ("rl_training.runtime.collector", "Collector"),
    "EarlyStoppingCallback": ("rl_training.runtime.controls", "EarlyStoppingCallback"),
    "EarlyStoppingConfig": ("rl_training.runtime.controls", "EarlyStoppingConfig"),
    "EvalResult": ("rl_training.runtime.evaluator", "EvalResult"),
    "Evaluator": ("rl_training.runtime.evaluator", "Evaluator"),
    "EvaluationRunner": ("rl_training.runtime.evaluation_runner", "EvaluationRunner"),
    "FunctionRunner": ("rl_training.runtime.runner", "FunctionRunner"),
    "LocalAsyncBackend": ("rl_training.runtime.vector_envs", "LocalAsyncBackend"),
    "LocalSyncBackend": ("rl_training.runtime.vector_envs", "LocalSyncBackend"),
    "Runner": ("rl_training.runtime.runner", "Runner"),
    "ScheduleSpec": ("rl_training.runtime.schedules", "ScheduleSpec"),
    "TrainResult": ("rl_training.runtime.trainer", "TrainResult"),
    "Trainer": ("rl_training.runtime.trainer", "Trainer"),
    "TrainerState": ("rl_training.runtime.trainer", "TrainerState"),
    "TrainingSession": ("rl_training.runtime.session", "TrainingSession"),
    "WorkerBackend": ("rl_training.runtime.vector_envs", "WorkerBackend"),
    "create_training_session": ("rl_training.runtime.session", "create_training_session"),
    "resolve_worker_backend": ("rl_training.runtime.vector_envs", "resolve_worker_backend"),
    "supported_execution_backends": ("rl_training.runtime.vector_envs", "supported_execution_backends"),
}


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, export_name = spec
    module = import_module(module_name)
    value = getattr(module, export_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)

