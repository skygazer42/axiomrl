import warnings
from importlib import import_module

from axiomrl.version import __version__

_STABLE_ROOT_EXPORTS: dict[str, tuple[str, str]] = {
    "STABLE_ALGORITHMS": ("axiomrl.core", "STABLE_ALGORITHMS"),
    "A2C": ("axiomrl.core", "A2C"),
    "BC": ("axiomrl.core", "BC"),
    "CQL": ("axiomrl.core", "CQL"),
    "DQN": ("axiomrl.core", "DQN"),
    "DiscreteSAC": ("axiomrl.core", "DiscreteSAC"),
    "IQL": ("axiomrl.core", "IQL"),
    "PPO": ("axiomrl.core", "PPO"),
    "SAC": ("axiomrl.core", "SAC"),
    "TD3": ("axiomrl.core", "TD3"),
    "TRPO": ("axiomrl.core", "TRPO"),
    "TrainConfig": ("axiomrl.core", "TrainConfig"),
}


def __getattr__(name: str):
    if name == "core":
        return import_module("axiomrl.core")
    if name == "experimental":
        return import_module("axiomrl.experimental")
    if name == "contrib":
        return import_module("axiomrl.contrib")
    if name in _STABLE_ROOT_EXPORTS:
        module_name, export_name = _STABLE_ROOT_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, export_name)
        globals()[name] = value
        return value
    api_module = import_module("axiomrl.api")
    if hasattr(api_module, name):
        warnings.warn(
            (
                f"axiomrl.{name} is no longer part of the stable root API and is deprecated; "
                "import advanced algorithms from axiomrl.experimental or axiomrl.api instead."
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
