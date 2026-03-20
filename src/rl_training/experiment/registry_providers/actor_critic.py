from __future__ import annotations

from rl_training.experiment.registry_core import _ALGORITHM_REGISTRY


_ACTOR_CRITIC_NAMES = (
    "sac",
    "rlpd",
    "crossq",
    "discrete_sac",
    "tqc",
    "redq",
    "edac",
    "ddpg",
    "naf",
    "d4pg",
    "drq",
    "curl",
    "drqv2",
    "td3",
    "td3_bc",
)

ACTOR_CRITIC_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _ACTOR_CRITIC_NAMES}
