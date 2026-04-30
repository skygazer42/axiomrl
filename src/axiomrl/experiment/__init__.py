from __future__ import annotations

from importlib import import_module


_EXPORTS: dict[str, tuple[str, str]] = {
    "AlgorithmSpec": ("rl_training.experiment.registry", "AlgorithmSpec"),
    "CheckpointState": ("rl_training.experiment.checkpointing", "CheckpointState"),
    "DefaultExperimentManager": ("rl_training.experiment.default_manager", "DefaultExperimentManager"),
    "JsonlLogger": ("rl_training.experiment.logging", "JsonlLogger"),
    "Logger": ("rl_training.experiment.logging", "Logger"),
    "RunLogger": ("rl_training.experiment.logging", "RunLogger"),
    "RunContext": ("rl_training.experiment.runs", "RunContext"),
    "TrainConfig": ("rl_training.experiment.config", "TrainConfig"),
    "create_run_context": ("rl_training.experiment.runs", "create_run_context"),
    "get_algorithm_spec": ("rl_training.experiment.registry", "get_algorithm_spec"),
    "list_algorithm_specs": ("rl_training.experiment.registry", "list_algorithm_specs"),
    "load_checkpoint": ("rl_training.experiment.checkpointing", "load_checkpoint"),
    "save_checkpoint": ("rl_training.experiment.checkpointing", "save_checkpoint"),
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

