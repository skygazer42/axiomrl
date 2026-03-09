from rl_training.algorithms.base import Algorithm, UpdateResult
from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.a2c import a2c_loss
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import dqn_loss
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.ppo import ppo_loss
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.sac import sac_loss
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3 import td3_loss

__all__ = [
    "Algorithm",
    "A2CAlgorithm",
    "DQN",
    "DQNAlgorithm",
    "PPO",
    "PPOAlgorithm",
    "SAC",
    "SACAlgorithm",
    "TD3",
    "TD3Algorithm",
    "UpdateResult",
    "a2c_loss",
    "dqn_loss",
    "ppo_loss",
    "sac_loss",
    "td3_loss",
]


def __getattr__(name: str):
    if name in {"A2C", "PPO", "DQN", "SAC", "TD3"}:
        from rl_training.api import A2C, DQN, PPO, SAC, TD3

        mapping = {
            "A2C": A2C,
            "PPO": PPO,
            "DQN": DQN,
            "SAC": SAC,
            "TD3": TD3,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
