"""Expanded managed-algorithm API for advanced and experimental workflows."""

from rl_training.api import *  # noqa: F401,F403
from rl_training.api import __all__ as _api_all
from rl_training.core import STABLE_ALGORITHMS


EXPERIMENTAL_ALGORITHMS = tuple(
    name for name in _api_all if name not in STABLE_ALGORITHMS and name != "contrib"
)


__all__ = list(_api_all) + ["EXPERIMENTAL_ALGORITHMS"]
