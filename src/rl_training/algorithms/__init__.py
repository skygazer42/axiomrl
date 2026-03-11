from rl_training.algorithms.base import Algorithm, UpdateResult
from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.a2c import a2c_loss
from rl_training.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from rl_training.algorithms.c51_dqn import c51_loss
from rl_training.algorithms.cql import CQL as CQLAlgorithm
from rl_training.algorithms.cql import cql_loss
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.ddpg import ddpg_loss
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.iql import iql_loss
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
from rl_training.algorithms.redq import REDQ as REDQAlgorithm
from rl_training.algorithms.redq import redq_loss
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.sac import sac_loss
from rl_training.algorithms.tqc import TQC as TQCAlgorithm
from rl_training.algorithms.tqc import tqc_loss
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3 import td3_loss
from rl_training.algorithms.td3_bc import TD3BC as TD3BCAlgorithm
from rl_training.algorithms.td3_bc import td3_bc_loss

__all__ = [
    "Algorithm",
    "A2CAlgorithm",
    "C51DQN",
    "C51DQNAlgorithm",
    "CQL",
    "CQLAlgorithm",
    "DDPG",
    "DDPGAlgorithm",
    "DQN",
    "DQNAlgorithm",
    "DoubleDQN",
    "DoubleDQNAlgorithm",
    "DuelingDQN",
    "DuelingDQNAlgorithm",
    "IQL",
    "IQLAlgorithm",
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
    "REDQ",
    "REDQAlgorithm",
    "PPO",
    "PPOAlgorithm",
    "SAC",
    "SACAlgorithm",
    "TQC",
    "TQCAlgorithm",
    "TD3",
    "TD3Algorithm",
    "TD3BC",
    "TD3BCAlgorithm",
    "UpdateResult",
    "a2c_loss",
    "c51_loss",
    "cql_loss",
    "ddpg_loss",
    "dqn_loss",
    "iql_loss",
    "iqn_loss",
    "ppo_loss",
    "qr_loss",
    "redq_loss",
    "sac_loss",
    "tqc_loss",
    "td3_loss",
    "td3_bc_loss",
]


def __getattr__(name: str):
    if name in {
        "A2C",
        "C51DQN",
        "CQL",
        "DDPG",
        "PPO",
        "DQN",
        "DoubleDQN",
        "DuelingDQN",
        "IQL",
        "IQN",
        "NoisyDQN",
        "NStepDQN",
        "PrioritizedDQN",
        "QRDQN",
        "RainbowDQN",
        "REDQ",
        "SAC",
        "TQC",
        "TD3",
        "TD3BC",
    }:
        from rl_training.api import (
            A2C,
            C51DQN,
            CQL,
            DDPG,
            DQN,
            DoubleDQN,
            DuelingDQN,
            IQL,
            IQN,
            NoisyDQN,
            NStepDQN,
            PPO,
            PrioritizedDQN,
            QRDQN,
            RainbowDQN,
            REDQ,
            SAC,
            TQC,
            TD3,
            TD3BC,
        )

        mapping = {
            "A2C": A2C,
            "C51DQN": C51DQN,
            "CQL": CQL,
            "DDPG": DDPG,
            "PPO": PPO,
            "DQN": DQN,
            "DoubleDQN": DoubleDQN,
            "DuelingDQN": DuelingDQN,
            "IQL": IQL,
            "IQN": IQN,
            "NoisyDQN": NoisyDQN,
            "NStepDQN": NStepDQN,
            "PrioritizedDQN": PrioritizedDQN,
            "QRDQN": QRDQN,
            "RainbowDQN": RainbowDQN,
            "REDQ": REDQ,
            "SAC": SAC,
            "TQC": TQC,
            "TD3": TD3,
            "TD3BC": TD3BC,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
