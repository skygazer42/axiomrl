"""Expanded managed-algorithm API for advanced and experimental workflows."""

from axiomrl.api import *  # noqa: F401,F403
from axiomrl.api import __all__ as _api_all
from axiomrl.core import STABLE_ALGORITHMS

EXPERIMENTAL_ALGORITHMS = tuple(name for name in _api_all if name not in STABLE_ALGORITHMS and name != "contrib")


__all__ = list(_api_all) + ["EXPERIMENTAL_ALGORITHMS"]
