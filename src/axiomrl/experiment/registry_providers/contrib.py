from __future__ import annotations

from axiomrl.experiment.registry_core import _ALGORITHM_REGISTRY

_CONTRIB_NAMES = ("recurrent_ppo",)

CONTRIB_SPECS = {name: _ALGORITHM_REGISTRY[name] for name in _CONTRIB_NAMES}
