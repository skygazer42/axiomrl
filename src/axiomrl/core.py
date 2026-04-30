"""Stable core API for application engineers.

This module defines the semver-governed public surface that AxiomRL commits to
keeping stable across 1.x releases.
"""

from rl_training.api import A2C, BC, CQL, DQN, DiscreteSAC, IQL, PPO, SAC, TD3, TRPO
from rl_training.experiment.config import TrainConfig


STABLE_ALGORITHMS = (
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
)


__all__ = [
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
]
