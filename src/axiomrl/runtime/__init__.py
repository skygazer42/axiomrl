from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "CollectResult": ("axiomrl.runtime.collector", "CollectResult"),
    "Collector": ("axiomrl.runtime.collector", "Collector"),
    "EarlyStoppingCallback": ("axiomrl.runtime.controls", "EarlyStoppingCallback"),
    "EarlyStoppingConfig": ("axiomrl.runtime.controls", "EarlyStoppingConfig"),
    "EvalResult": ("axiomrl.runtime.evaluator", "EvalResult"),
    "Evaluator": ("axiomrl.runtime.evaluator", "Evaluator"),
    "EvaluationRunner": ("axiomrl.runtime.evaluation_runner", "EvaluationRunner"),
    "FunctionRunner": ("axiomrl.runtime.runner", "FunctionRunner"),
    "LocalAsyncBackend": ("axiomrl.runtime.vector_envs", "LocalAsyncBackend"),
    "LocalSyncBackend": ("axiomrl.runtime.vector_envs", "LocalSyncBackend"),
    "Runner": ("axiomrl.runtime.runner", "Runner"),
    "ScheduleSpec": ("axiomrl.runtime.schedules", "ScheduleSpec"),
    "TrainResult": ("axiomrl.runtime.trainer", "TrainResult"),
    "Trainer": ("axiomrl.runtime.trainer", "Trainer"),
    "TrainerState": ("axiomrl.runtime.trainer", "TrainerState"),
    "TrainingSession": ("axiomrl.runtime.session", "TrainingSession"),
    "WorkerBackend": ("axiomrl.runtime.vector_envs", "WorkerBackend"),
    "create_training_session": ("axiomrl.runtime.session", "create_training_session"),
    "resolve_worker_backend": ("axiomrl.runtime.vector_envs", "resolve_worker_backend"),
    "supported_execution_backends": ("axiomrl.runtime.vector_envs", "supported_execution_backends"),
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
