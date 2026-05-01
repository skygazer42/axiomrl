from axiomrl.experiment.registry_core import _ALGORITHM_REGISTRY

_GOAL_CONDITIONED_NAMES = ("her",)

GOAL_CONDITIONED_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _GOAL_CONDITIONED_NAMES}
