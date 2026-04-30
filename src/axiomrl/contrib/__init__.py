from __future__ import annotations


__all__ = ["RecurrentPPO", "RecurrentPPOAlgorithm"]


def __getattr__(name: str):
    if name == "RecurrentPPO":
        from rl_training.contrib.api import RecurrentPPO

        return RecurrentPPO
    if name == "RecurrentPPOAlgorithm":
        from rl_training.contrib.recurrent_ppo import RecurrentPPOAlgorithm

        return RecurrentPPOAlgorithm
    raise AttributeError(name)
