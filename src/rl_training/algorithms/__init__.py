from rl_training.algorithms.base import Algorithm, UpdateResult
from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.a2c import a2c_loss
from rl_training.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from rl_training.algorithms.c51_dqn import c51_loss
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.ddpg import ddpg_loss
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.iqn import IQN as IQNAlgorithm
from rl_training.algorithms.iqn import iqn_loss
from rl_training.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from rl_training.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from rl_training.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from rl_training.algorithms.dqn import dqn_loss
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.ppo import ppo_loss
from rl_training.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from rl_training.algorithms.qr_dqn import qr_loss
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.sac import sac_loss
from rl_training.algorithms.tqc import TQC as TQCAlgorithm
from rl_training.algorithms.tqc import tqc_loss
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3 import td3_loss

__all__ = [
    "Algorithm",
    "A2CAlgorithm",
    "C51DQN",
    "C51DQNAlgorithm",
    "DDPG",
    "DDPGAlgorithm",
    "DQN",
    "DQNAlgorithm",
    "DoubleDQN",
    "DoubleDQNAlgorithm",
    "DuelingDQN",
    "DuelingDQNAlgorithm",
    "IQN",
    "IQNAlgorithm",
    "NoisyDQN",
    "NoisyDQNAlgorithm",
    "NStepDQN",
    "PrioritizedDQN",
    "PrioritizedDQNAlgorithm",
    "QRDQN",
    "QRDQNAlgorithm",
    "RainbowDQN",
    "RainbowDQNAlgorithm",
    "PPO",
    "PPOAlgorithm",
    "SAC",
    "SACAlgorithm",
    "TQC",
    "TQCAlgorithm",
    "TD3",
    "TD3Algorithm",
    "UpdateResult",
    "a2c_loss",
    "c51_loss",
    "ddpg_loss",
    "dqn_loss",
    "iqn_loss",
    "ppo_loss",
    "qr_loss",
    "sac_loss",
    "tqc_loss",
    "td3_loss",
]


def __getattr__(name: str):
    if name in {
        "A2C",
        "C51DQN",
        "DDPG",
        "PPO",
        "DQN",
        "DoubleDQN",
        "DuelingDQN",
        "IQN",
        "NoisyDQN",
        "NStepDQN",
        "PrioritizedDQN",
        "QRDQN",
        "RainbowDQN",
        "SAC",
        "TQC",
        "TD3",
    }:
        from rl_training.api import (
            A2C,
            C51DQN,
            DDPG,
            DQN,
            DoubleDQN,
            DuelingDQN,
            IQN,
            NoisyDQN,
            NStepDQN,
            PPO,
            PrioritizedDQN,
            QRDQN,
            RainbowDQN,
            SAC,
            TQC,
            TD3,
        )

        mapping = {
            "A2C": A2C,
            "C51DQN": C51DQN,
            "DDPG": DDPG,
            "PPO": PPO,
            "DQN": DQN,
            "DoubleDQN": DoubleDQN,
            "DuelingDQN": DuelingDQN,
            "IQN": IQN,
            "NoisyDQN": NoisyDQN,
            "NStepDQN": NStepDQN,
            "PrioritizedDQN": PrioritizedDQN,
            "QRDQN": QRDQN,
            "RainbowDQN": RainbowDQN,
            "SAC": SAC,
            "TQC": TQC,
            "TD3": TD3,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
