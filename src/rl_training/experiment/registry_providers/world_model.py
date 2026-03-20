from __future__ import annotations

from rl_training.experiment.registry_core import _ALGORITHM_REGISTRY


_WORLD_MODEL_NAMES = (
    "dreamer",
    "dreamerv3",
    "diamond",
    "horizon_imagination",
    "po_dreamer",
    "twisted",
    "mow",
    "eadream",
    "muzero",
    "gumbel_muzero",
    "efficientzero",
    "scalezero",
    "mopo",
    "mbpo",
    "pets",
)

WORLD_MODEL_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _WORLD_MODEL_NAMES}
