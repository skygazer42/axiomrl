from __future__ import annotations

__all__ = ["RecurrentPPO", "RecurrentPPOAlgorithm"]


def __getattr__(name: str):
    if name == "RecurrentPPO":
        from axiomrl.contrib.api import RecurrentPPO

        return RecurrentPPO
    if name == "RecurrentPPOAlgorithm":
        from axiomrl.contrib.recurrent_ppo import RecurrentPPOAlgorithm

        return RecurrentPPOAlgorithm
    raise AttributeError(name)
