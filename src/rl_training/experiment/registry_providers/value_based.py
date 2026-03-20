from __future__ import annotations

from rl_training.experiment.registry_core import _ALGORITHM_REGISTRY


_VALUE_BASED_NAMES = (
    "dqn",
    "jowa",
    "spr",
    "apex_dqn",
    "c51_dqn",
    "n_step_dqn",
    "expected_sarsa",
    "expected_double_dqn",
    "boltzmann_dqn",
    "boltzmann_double_dqn",
    "mellowmax_dqn",
    "soft_dqn",
    "soft_double_dqn",
    "advantage_learning_dqn",
    "persistent_advantage_learning_dqn",
    "munchausen_dqn",
    "munchausen_double_dqn",
    "cql_dqn",
    "cql_double_dqn",
    "clipped_double_dqn",
    "hysteretic_dqn",
    "noisy_dqn",
    "prioritized_dqn",
    "rainbow_dqn",
    "qr_dqn",
    "iqn",
    "fqf",
    "double_dqn",
    "dueling_dqn",
    "drqn",
    "r2d2",
    "agent57",
)

VALUE_BASED_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _VALUE_BASED_NAMES}
