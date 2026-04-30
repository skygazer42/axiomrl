from importlib import import_module
import warnings

from rl_training.version import __version__

_STABLE_ROOT_EXPORTS: dict[str, tuple[str, str]] = {
    "STABLE_ALGORITHMS": ("rl_training.core", "STABLE_ALGORITHMS"),
    "A2C": ("rl_training.core", "A2C"),
    "BC": ("rl_training.core", "BC"),
    "CQL": ("rl_training.core", "CQL"),
    "DQN": ("rl_training.core", "DQN"),
    "DiscreteSAC": ("rl_training.core", "DiscreteSAC"),
    "IQL": ("rl_training.core", "IQL"),
    "PPO": ("rl_training.core", "PPO"),
    "SAC": ("rl_training.core", "SAC"),
    "TD3": ("rl_training.core", "TD3"),
    "TRPO": ("rl_training.core", "TRPO"),
    "TrainConfig": ("rl_training.core", "TrainConfig"),
}


def __getattr__(name: str):
    if name == "core":
        return import_module("rl_training.core")
    if name == "experimental":
        return import_module("rl_training.experimental")
    if name == "contrib":
        return import_module("rl_training.contrib")
    if name in _STABLE_ROOT_EXPORTS:
        module_name, export_name = _STABLE_ROOT_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, export_name)
        globals()[name] = value
        return value
    api_module = import_module("rl_training.api")
    if hasattr(api_module, name):
        warnings.warn(
            (
                f"rl_training.{name} is no longer part of the stable root API and is deprecated; "
                "import advanced algorithms from rl_training.experimental or rl_training.api instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        value = getattr(api_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "STABLE_ALGORITHMS",
    "A2C",
    "BC",
    "CQL",
    "DQN",
    "DiscreteSAC",
    "IQL",
    "PPO",
    "SAC",
    "TD3",
    "TRPO",
    "TrainConfig",
    "core",
    "experimental",
    "contrib",
]
