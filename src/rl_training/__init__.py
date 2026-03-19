from importlib import import_module
import warnings

from rl_training.core import A2C, BC, CQL, DQN, DiscreteSAC, IQL, PPO, SAC, STABLE_ALGORITHMS, TD3, TRPO, TrainConfig
from rl_training.version import __version__


def __getattr__(name: str):
    if name == "core":
        return import_module("rl_training.core")
    if name == "experimental":
        return import_module("rl_training.experimental")
    if name == "contrib":
        return import_module("rl_training.contrib")
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
        return getattr(api_module, name)
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
