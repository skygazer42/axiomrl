from rl_training.algorithms.ars import ARS as ARSAlgorithm
from rl_training.algorithms.ars import ars_loss
from rl_training.algorithms.openai_es import OpenAIES as OpenAIESAlgorithm
from rl_training.algorithms.openai_es import openai_es_loss
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
from rl_training.algorithms.impala import IMPALA as IMPALAAlgorithm
from rl_training.algorithms.impala import impala_loss
from rl_training.algorithms.appo import APPO as APPOAlgorithm
from rl_training.algorithms.appo import appo_loss
from rl_training.algorithms.decision_transformer import DecisionTransformer as DecisionTransformerAlgorithm
from rl_training.algorithms.decision_transformer import decision_transformer_loss
from rl_training.algorithms.mopo import MOPO as MOPOAlgorithm
from rl_training.algorithms.mopo import mopo_model_loss
from rl_training.algorithms.pets import PETS as PETSAlgorithm
from rl_training.algorithms.pets import pets_loss
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
from rl_training.algorithms.curl import CURL as CURLAlgorithm
from rl_training.algorithms.curl import curl_loss
from rl_training.algorithms.d4pg import D4PG as D4PGAlgorithm
from rl_training.algorithms.d4pg import d4pg_loss
from rl_training.algorithms.drqn import DRQN as DRQNAlgorithm
from rl_training.algorithms.drqn import drqn_loss
from rl_training.algorithms.r2d2 import R2D2 as R2D2Algorithm
from rl_training.algorithms.r2d2 import r2d2_loss
from rl_training.algorithms.naf import NAF as NAFAlgorithm
from rl_training.algorithms.naf import naf_loss
from rl_training.algorithms.ddpg import DDPG as DDPGAlgorithm
from rl_training.algorithms.ddpg import ddpg_loss
from rl_training.algorithms.edac import EDAC as EDACAlgorithm
from rl_training.algorithms.edac import critic_diversity_loss
from rl_training.algorithms.edac import edac_loss
from rl_training.algorithms.drq import DrQ as DrQAlgorithm
from rl_training.algorithms.drq import drq_loss
from rl_training.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from rl_training.algorithms.drqv2 import drqv2_loss
from rl_training.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from rl_training.algorithms.discrete_sac import discrete_sac_loss
from rl_training.algorithms.efficientzero import EfficientZero as EfficientZeroAlgorithm
from rl_training.algorithms.gumbel_muzero import GumbelMuZero as GumbelMuZeroAlgorithm
from rl_training.algorithms.ppg import PPG as PPGAlgorithm
from rl_training.algorithms.ppg import ppg_auxiliary_loss
from rl_training.algorithms.ppg import ppg_loss
from rl_training.algorithms.dqn import DQN as DQNAlgorithm
from rl_training.algorithms.dqn import AdvantageLearningDQN as AdvantageLearningDQNAlgorithm
from rl_training.algorithms.dqn import BoltzmannDQN as BoltzmannDQNAlgorithm
from rl_training.algorithms.dqn import BoltzmannDoubleDQN as BoltzmannDoubleDQNAlgorithm
from rl_training.algorithms.dqn import CQLDQN as CQLDQNAlgorithm
from rl_training.algorithms.dqn import CQLDoubleDQN as CQLDoubleDQNAlgorithm
from rl_training.algorithms.dqn import ClippedDoubleDQN as ClippedDoubleDQNAlgorithm
from rl_training.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from rl_training.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from rl_training.algorithms.dqn import ExpectedDoubleDQN as ExpectedDoubleDQNAlgorithm
from rl_training.algorithms.dqn import ExpectedSARSA as ExpectedSARSAAlgorithm
from rl_training.algorithms.her import HER as HERAlgorithm
from rl_training.algorithms.her import her_loss
from rl_training.algorithms.iql import IQL as IQLAlgorithm
from rl_training.algorithms.iql import iql_loss
from rl_training.algorithms.iqn import IQN as IQNAlgorithm
from rl_training.algorithms.iqn import iqn_loss
from rl_training.algorithms.dqn import HystereticDQN as HystereticDQNAlgorithm
from rl_training.algorithms.dqn import MunchausenDoubleDQN as MunchausenDoubleDQNAlgorithm
from rl_training.algorithms.dqn import PersistentAdvantageLearningDQN as PersistentAdvantageLearningDQNAlgorithm
from rl_training.algorithms.dqn import SoftDoubleDQN as SoftDoubleDQNAlgorithm
from rl_training.algorithms.xql import XQL as XQLAlgorithm
from rl_training.algorithms.xql import gumbel_rescale_loss
from rl_training.algorithms.xql import xql_loss
from rl_training.algorithms.xql import xql_value_loss
from rl_training.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from rl_training.algorithms.dqn import MunchausenDQN as MunchausenDQNAlgorithm
from rl_training.algorithms.dqn import MellowmaxDQN as MellowmaxDQNAlgorithm
from rl_training.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from rl_training.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from rl_training.algorithms.dqn import SoftDQN as SoftDQNAlgorithm
from rl_training.algorithms.dqn import dqn_loss
from rl_training.algorithms.ppo import PPO as PPOAlgorithm
from rl_training.algorithms.ppo import ppo_loss
from rl_training.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from rl_training.algorithms.qr_dqn import qr_loss
from rl_training.algorithms.spr import SPR as SPRAlgorithm
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
    "ARS",
    "ARSAlgorithm",
    "OpenAIES",
    "OpenAIESAlgorithm",
    "A2CAlgorithm",
    "AWR",
    "AWRAlgorithm",
    "AWAC",
    "AWACAlgorithm",
    "MARWIL",
    "MARWILAlgorithm",
    "BC",
    "BCAlgorithm",
    "IMPALA",
    "IMPALAAlgorithm",
    "APPO",
    "APPOAlgorithm",
    "DecisionTransformer",
    "DecisionTransformerAlgorithm",
    "MOPO",
    "MOPOAlgorithm",
    "PETS",
    "PETSAlgorithm",
    "BCQ",
    "BCQAlgorithm",
    "BEAR",
    "BEARAlgorithm",
    "C51DQN",
    "C51DQNAlgorithm",
    "CalQL",
    "CalQLAlgorithm",
    "CURL",
    "CURLAlgorithm",
    "CrossQ",
    "CrossQAlgorithm",
    "CRR",
    "CRRAlgorithm",
    "CQL",
    "CQLAlgorithm",
    "D4PG",
    "D4PGAlgorithm",
    "DRQN",
    "DRQNAlgorithm",
    "R2D2",
    "R2D2Algorithm",
    "NAF",
    "NAFAlgorithm",
    "DDPG",
    "DDPGAlgorithm",
    "EDAC",
    "EDACAlgorithm",
    "DrQ",
    "DrQAlgorithm",
    "DrQv2",
    "DrQv2Algorithm",
    "DiscreteSAC",
    "DiscreteSACAlgorithm",
    "EfficientZero",
    "EfficientZeroAlgorithm",
    "GumbelMuZero",
    "GumbelMuZeroAlgorithm",
    "PPG",
    "PPGAlgorithm",
    "DQN",
    "DQNAlgorithm",
    "AdvantageLearningDQN",
    "AdvantageLearningDQNAlgorithm",
    "BoltzmannDoubleDQN",
    "BoltzmannDoubleDQNAlgorithm",
    "BoltzmannDQN",
    "BoltzmannDQNAlgorithm",
    "CQLDQN",
    "CQLDQNAlgorithm",
    "CQLDoubleDQN",
    "CQLDoubleDQNAlgorithm",
    "ClippedDoubleDQN",
    "ClippedDoubleDQNAlgorithm",
    "ExpectedDoubleDQN",
    "ExpectedDoubleDQNAlgorithm",
    "ExpectedSARSA",
    "ExpectedSARSAAlgorithm",
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
    "HystereticDQN",
    "HystereticDQNAlgorithm",
    "MunchausenDQN",
    "MunchausenDQNAlgorithm",
    "MunchausenDoubleDQN",
    "MunchausenDoubleDQNAlgorithm",
    "MellowmaxDQN",
    "MellowmaxDQNAlgorithm",
    "PersistentAdvantageLearningDQN",
    "PersistentAdvantageLearningDQNAlgorithm",
    "SoftDQN",
    "SoftDQNAlgorithm",
    "SoftDoubleDQN",
    "SoftDoubleDQNAlgorithm",
    "XQL",
    "XQLAlgorithm",
    "NoisyDQN",
    "NoisyDQNAlgorithm",
    "NStepDQN",
    "PrioritizedDQN",
    "PrioritizedDQNAlgorithm",
    "QRDQN",
    "QRDQNAlgorithm",
    "SPR",
    "SPRAlgorithm",
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
    "ars_loss",
    "openai_es_loss",
    "a2c_loss",
    "awr_loss",
    "awac_loss",
    "marwil_loss",
    "bc_loss",
    "impala_loss",
    "appo_loss",
    "decision_transformer_loss",
    "mopo_model_loss",
    "pets_loss",
    "bcq_loss",
    "bear_loss",
    "c51_loss",
    "cal_ql_loss",
    "curl_loss",
    "crossq_loss",
    "crr_loss",
    "cql_loss",
    "d4pg_loss",
    "drqn_loss",
    "r2d2_loss",
    "naf_loss",
    "ddpg_loss",
    "critic_diversity_loss",
    "edac_loss",
    "drq_loss",
    "drqv2_loss",
    "discrete_sac_loss",
    "ppg_loss",
    "ppg_auxiliary_loss",
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
        "ARS",
        "OpenAIES",
        "A2C",
        "AWR",
        "AWAC",
        "MARWIL",
        "BC",
        "IMPALA",
        "APPO",
        "DecisionTransformer",
        "MOPO",
        "PETS",
        "BCQ",
        "BEAR",
        "C51DQN",
        "CalQL",
        "CURL",
        "CrossQ",
        "CRR",
        "CQL",
        "D4PG",
        "DRQN",
        "R2D2",
        "NAF",
        "DDPG",
        "EDAC",
        "DrQ",
        "DrQv2",
        "DiscreteSAC",
        "EfficientZero",
        "GumbelMuZero",
        "PPG",
        "PPO",
        "DQN",
        "AdvantageLearningDQN",
        "BoltzmannDoubleDQN",
        "BoltzmannDQN",
        "CQLDQN",
        "CQLDoubleDQN",
        "ClippedDoubleDQN",
        "ExpectedDoubleDQN",
        "ExpectedSARSA",
        "HystereticDQN",
        "MellowmaxDQN",
        "MunchausenDoubleDQN",
        "PersistentAdvantageLearningDQN",
        "SoftDQN",
        "SoftDoubleDQN",
        "DoubleDQN",
        "DuelingDQN",
        "HER",
        "IQL",
        "IQN",
        "MunchausenDQN",
        "XQL",
        "NoisyDQN",
        "NStepDQN",
        "PrioritizedDQN",
        "QRDQN",
        "SPR",
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
            ARS,
            OpenAIES,
            A2C,
            AWR,
            AWAC,
            MARWIL,
            BC,
            IMPALA,
            APPO,
            DecisionTransformer,
            MOPO,
            PETS,
            BCQ,
            BEAR,
            C51DQN,
            CalQL,
            CURL,
            CrossQ,
            CRR,
            CQL,
            D4PG,
            DRQN,
            R2D2,
            NAF,
            DDPG,
            EDAC,
            DrQ,
            DrQv2,
            DiscreteSAC,
            EfficientZero,
            GumbelMuZero,
            PPG,
            DQN,
            AdvantageLearningDQN,
            BoltzmannDoubleDQN,
            BoltzmannDQN,
            CQLDQN,
            CQLDoubleDQN,
            ClippedDoubleDQN,
            ExpectedDoubleDQN,
            ExpectedSARSA,
            HystereticDQN,
            MellowmaxDQN,
            MunchausenDoubleDQN,
            PersistentAdvantageLearningDQN,
            SoftDQN,
            SoftDoubleDQN,
            DoubleDQN,
            DuelingDQN,
            HER,
            IQL,
            IQN,
            MunchausenDQN,
            XQL,
            NoisyDQN,
            NStepDQN,
            PPO,
            PrioritizedDQN,
            QRDQN,
            SPR,
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
            "ARS": ARS,
            "OpenAIES": OpenAIES,
            "A2C": A2C,
            "AWR": AWR,
            "AWAC": AWAC,
            "MARWIL": MARWIL,
            "BC": BC,
            "IMPALA": IMPALA,
            "APPO": APPO,
            "DecisionTransformer": DecisionTransformer,
            "MOPO": MOPO,
            "PETS": PETS,
            "BCQ": BCQ,
            "BEAR": BEAR,
            "C51DQN": C51DQN,
            "CalQL": CalQL,
            "CURL": CURL,
            "CrossQ": CrossQ,
            "CRR": CRR,
            "CQL": CQL,
            "D4PG": D4PG,
            "DRQN": DRQN,
            "R2D2": R2D2,
            "NAF": NAF,
            "DDPG": DDPG,
            "EDAC": EDAC,
            "DrQ": DrQ,
            "DrQv2": DrQv2,
            "DiscreteSAC": DiscreteSAC,
            "EfficientZero": EfficientZero,
            "GumbelMuZero": GumbelMuZero,
            "PPG": PPG,
            "PPO": PPO,
            "DQN": DQN,
            "AdvantageLearningDQN": AdvantageLearningDQN,
            "BoltzmannDoubleDQN": BoltzmannDoubleDQN,
            "BoltzmannDQN": BoltzmannDQN,
            "CQLDQN": CQLDQN,
            "CQLDoubleDQN": CQLDoubleDQN,
            "ClippedDoubleDQN": ClippedDoubleDQN,
            "ExpectedDoubleDQN": ExpectedDoubleDQN,
            "ExpectedSARSA": ExpectedSARSA,
            "HystereticDQN": HystereticDQN,
            "MellowmaxDQN": MellowmaxDQN,
            "MunchausenDoubleDQN": MunchausenDoubleDQN,
            "PersistentAdvantageLearningDQN": PersistentAdvantageLearningDQN,
            "SoftDQN": SoftDQN,
            "SoftDoubleDQN": SoftDoubleDQN,
            "DoubleDQN": DoubleDQN,
            "DuelingDQN": DuelingDQN,
            "HER": HER,
            "IQL": IQL,
            "IQN": IQN,
            "MunchausenDQN": MunchausenDQN,
            "XQL": XQL,
            "NoisyDQN": NoisyDQN,
            "NStepDQN": NStepDQN,
            "PrioritizedDQN": PrioritizedDQN,
            "QRDQN": QRDQN,
            "SPR": SPR,
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
