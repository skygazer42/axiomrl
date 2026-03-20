from __future__ import annotations

from rl_training.experiment.registry_core import _ALGORITHM_REGISTRY


_ON_POLICY_NAMES = (
    "a2c",
    "ars",
    "openai_es",
    "impala",
    "appo",
    "ppo",
    "gail",
    "ppg",
    "trpo",
)

ON_POLICY_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _ON_POLICY_NAMES}
