from rl_training.algorithms.base import Algorithm, UpdateResult
from rl_training.algorithms.a2c import A2C as A2CAlgorithm
from rl_training.algorithms.a2c import a2c_loss
from rl_training.algorithms.awr import AWR as AWRAlgorithm
from rl_training.algorithms.awr import awr_loss
from rl_training.algorithms.awac import AWAC as AWACAlgorithm
from rl_training.algorithms.awac import awac_loss
from rl_training.algorithms.marwil import MARWIL as MARWILAlgorithm
from rl_training.algorithms.marwil import marwil_loss
from rl_training.algorithms.bc import BC as BCAlgorithm
from rl_training.algorithms.bc import bc_loss
from rl_training.algorithms.bcq import BCQ as BCQAlgorithm
from rl_training.algorithms.bcq import bcq_loss
from rl_training.algorithms.bear import BEAR as BEARAlgorithm
from rl_training.algorithms.bear import bear_loss
from rl_training.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from rl_training.algorithms.c51_dqn import c51_loss
from rl_training.algorithms.crossq import CrossQ as CrossQAlgorithm
from rl_training.algorithms.crossq import crossq_loss
from rl_training.algorithms.crr import CRR as CRRAlgorithm
from rl_training.algorithms.crr import crr_loss
from rl_training.algorithms.cal_ql import CalQL as CalQLAlgorithm
from rl_training.algorithms.cal_ql import cal_ql_loss
from rl_training.algorithms.cql import CQL as CQLAlgorithm
from rl_training.algorithms.cql import cql_loss
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.ddpg import ddpg_loss
from rl_training.algorithms.edac import EDAC as EDACAlgorithm
from rl_training.algorithms.edac import critic_diversity_loss
from rl_training.algorithms.edac import edac_loss
from rl_training.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from rl_training.algorithms.drqv2 import drqv2_loss
from rl_training.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from rl_training.algorithms.discrete_sac import discrete_sac_loss
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.her import HER as HERAlgorithm
from rl_training.algorithms.her import her_loss
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.iql import iql_loss
from rl_training.algorithms.iqn import IQN as IQNAlgorithm
from rl_training.algorithms.iqn import iqn_loss
from rl_training.algorithms.xql import XQL as XQLAlgorithm
from rl_training.algorithms.xql import gumbel_rescale_loss
from rl_training.algorithms.xql import xql_loss
from rl_training.algorithms.xql import xql_value_loss
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
from rl_training.algorithms.rlpd import RLPD as RLPDAlgorithm
from rl_training.algorithms.rlpd import rlpd_loss
from rl_training.algorithms.rebrac import ReBRAC as ReBRACAlgorithm
from rl_training.algorithms.rebrac import rebrac_loss
from rl_training.algorithms.sac import SAC as SACAlgorithm
from rl_training.algorithms.sac import sac_loss
from rl_training.algorithms.trpo import TRPO as TRPOAlgorithm
from rl_training.algorithms.trpo import trpo_loss
from rl_training.algorithms.tqc import TQC as TQCAlgorithm
from rl_training.algorithms.tqc import tqc_loss
from rl_training.algorithms.td3 import TD3 as TD3Algorithm
from rl_training.algorithms.td3 import td3_loss
from rl_training.algorithms.td3_bc import TD3BC as TD3BCAlgorithm
from rl_training.algorithms.td3_bc import td3_bc_loss

__all__ = [
    "Algorithm",
    "A2CAlgorithm",
    "AWR",
    "AWRAlgorithm",
    "AWAC",
    "AWACAlgorithm",
    "MARWIL",
    "MARWILAlgorithm",
    "BC",
    "BCAlgorithm",
    "BCQ",
    "BCQAlgorithm",
    "BEAR",
    "BEARAlgorithm",
    "C51DQN",
    "C51DQNAlgorithm",
    "CalQL",
    "CalQLAlgorithm",
    "CrossQ",
    "CrossQAlgorithm",
    "CRR",
    "CRRAlgorithm",
    "CQL",
    "CQLAlgorithm",
    "DDPG",
    "DDPGAlgorithm",
    "EDAC",
    "EDACAlgorithm",
    "DrQv2",
    "DrQv2Algorithm",
    "DiscreteSAC",
    "DiscreteSACAlgorithm",
    "DQN",
    "DQNAlgorithm",
    "DoubleDQN",
    "DoubleDQNAlgorithm",
    "DuelingDQN",
    "DuelingDQNAlgorithm",
    "HER",
    "HERAlgorithm",
    "IQL",
    "IQLAlgorithm",
    "IQN",
    "IQNAlgorithm",
    "XQL",
    "XQLAlgorithm",
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
    "RLPD",
    "RLPDAlgorithm",
    "ReBRAC",
    "ReBRACAlgorithm",
    "PPO",
    "PPOAlgorithm",
    "SAC",
    "SACAlgorithm",
    "TRPO",
    "TRPOAlgorithm",
    "TQC",
    "TQCAlgorithm",
    "TD3",
    "TD3Algorithm",
    "TD3BC",
    "TD3BCAlgorithm",
    "UpdateResult",
    "a2c_loss",
    "awr_loss",
    "awac_loss",
    "marwil_loss",
    "bc_loss",
    "bcq_loss",
    "bear_loss",
    "c51_loss",
    "cal_ql_loss",
    "crossq_loss",
    "crr_loss",
    "cql_loss",
    "ddpg_loss",
    "critic_diversity_loss",
    "edac_loss",
    "drqv2_loss",
    "discrete_sac_loss",
    "dqn_loss",
    "her_loss",
    "iql_loss",
    "iqn_loss",
    "gumbel_rescale_loss",
    "ppo_loss",
    "qr_loss",
    "redq_loss",
    "rlpd_loss",
    "rebrac_loss",
    "sac_loss",
    "trpo_loss",
    "tqc_loss",
    "td3_loss",
    "td3_bc_loss",
    "xql_loss",
    "xql_value_loss",
]


def __getattr__(name: str):
    if name in {
        "A2C",
        "AWR",
        "AWAC",
        "MARWIL",
        "BC",
        "BCQ",
        "BEAR",
        "C51DQN",
        "CalQL",
        "CrossQ",
        "CRR",
        "CQL",
        "DDPG",
        "EDAC",
        "DrQv2",
        "DiscreteSAC",
        "PPO",
        "DQN",
        "DoubleDQN",
        "DuelingDQN",
        "HER",
        "IQL",
        "IQN",
        "XQL",
        "NoisyDQN",
        "NStepDQN",
        "PrioritizedDQN",
        "QRDQN",
        "RainbowDQN",
        "REDQ",
        "RLPD",
        "ReBRAC",
        "SAC",
        "TRPO",
        "TQC",
        "TD3",
        "TD3BC",
    }:
        from rl_training.api import (
            A2C,
            AWR,
            AWAC,
            MARWIL,
            BC,
            BCQ,
            BEAR,
            C51DQN,
            CalQL,
            CrossQ,
            CRR,
            CQL,
            DDPG,
            EDAC,
            DrQv2,
            DiscreteSAC,
            DQN,
            DoubleDQN,
            DuelingDQN,
            HER,
            IQL,
            IQN,
            XQL,
            NoisyDQN,
            NStepDQN,
            PPO,
            PrioritizedDQN,
            QRDQN,
            RainbowDQN,
            REDQ,
            RLPD,
            ReBRAC,
            SAC,
            TRPO,
            TQC,
            TD3,
            TD3BC,
        )

        mapping = {
            "A2C": A2C,
            "AWR": AWR,
            "AWAC": AWAC,
            "MARWIL": MARWIL,
            "BC": BC,
            "BCQ": BCQ,
            "BEAR": BEAR,
            "C51DQN": C51DQN,
            "CalQL": CalQL,
            "CrossQ": CrossQ,
            "CRR": CRR,
            "CQL": CQL,
            "DDPG": DDPG,
            "EDAC": EDAC,
            "DrQv2": DrQv2,
            "DiscreteSAC": DiscreteSAC,
            "PPO": PPO,
            "DQN": DQN,
            "DoubleDQN": DoubleDQN,
            "DuelingDQN": DuelingDQN,
            "HER": HER,
            "IQL": IQL,
            "IQN": IQN,
            "XQL": XQL,
            "NoisyDQN": NoisyDQN,
            "NStepDQN": NStepDQN,
            "PrioritizedDQN": PrioritizedDQN,
            "QRDQN": QRDQN,
            "RainbowDQN": RainbowDQN,
            "REDQ": REDQ,
            "RLPD": RLPD,
            "ReBRAC": ReBRAC,
            "SAC": SAC,
            "TRPO": TRPO,
            "TQC": TQC,
            "TD3": TD3,
            "TD3BC": TD3BC,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
