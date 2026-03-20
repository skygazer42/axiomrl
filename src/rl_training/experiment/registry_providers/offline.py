from __future__ import annotations

from rl_training.experiment.registry_core import _ALGORITHM_REGISTRY


_OFFLINE_NAMES = (
    "bc",
    "decision_transformer",
    "bcq",
    "bear",
    "awac",
    "crr",
    "cal_ql",
    "xql",
    "iql",
    "awr",
    "marwil",
    "cql",
    "rebrac",
)

OFFLINE_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _OFFLINE_NAMES}
