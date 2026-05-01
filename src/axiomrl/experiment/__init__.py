from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "AlgorithmSpec": ("axiomrl.experiment.registry", "AlgorithmSpec"),
    "CheckpointState": ("axiomrl.experiment.checkpointing", "CheckpointState"),
    "DefaultExperimentManager": ("axiomrl.experiment.default_manager", "DefaultExperimentManager"),
    "JsonlLogger": ("axiomrl.experiment.logging", "JsonlLogger"),
    "Logger": ("axiomrl.experiment.logging", "Logger"),
    "RunLogger": ("axiomrl.experiment.logging", "RunLogger"),
    "RunContext": ("axiomrl.experiment.runs", "RunContext"),
    "TrainConfig": ("axiomrl.experiment.config", "TrainConfig"),
    "create_run_context": ("axiomrl.experiment.runs", "create_run_context"),
    "get_algorithm_spec": ("axiomrl.experiment.registry", "get_algorithm_spec"),
    "list_algorithm_specs": ("axiomrl.experiment.registry", "list_algorithm_specs"),
    "load_checkpoint": ("axiomrl.experiment.checkpointing", "load_checkpoint"),
    "save_checkpoint": ("axiomrl.experiment.checkpointing", "save_checkpoint"),
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
