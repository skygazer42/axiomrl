from axiomrl.algorithms.a2c import A2C as A2CAlgorithm
from axiomrl.algorithms.a2c import a2c_loss
from axiomrl.algorithms.appo import APPO as APPOAlgorithm
from axiomrl.algorithms.appo import appo_loss
from axiomrl.algorithms.ars import ARS as ARSAlgorithm
from axiomrl.algorithms.ars import ars_loss
from axiomrl.algorithms.awac import AWAC as AWACAlgorithm
from axiomrl.algorithms.awac import awac_loss
from axiomrl.algorithms.awr import AWR as AWRAlgorithm
from axiomrl.algorithms.awr import awr_loss
from axiomrl.algorithms.base import Algorithm, UpdateResult
from axiomrl.algorithms.bc import BC as BCAlgorithm
from axiomrl.algorithms.bc import bc_loss
from axiomrl.algorithms.bcq import BCQ as BCQAlgorithm
from axiomrl.algorithms.bcq import bcq_loss
from axiomrl.algorithms.bear import BEAR as BEARAlgorithm
from axiomrl.algorithms.bear import bear_loss
from axiomrl.algorithms.c51_dqn import C51DQN as C51DQNAlgorithm
from axiomrl.algorithms.c51_dqn import c51_loss
from axiomrl.algorithms.cal_ql import CalQL as CalQLAlgorithm
from axiomrl.algorithms.cal_ql import cal_ql_loss
from axiomrl.algorithms.cql import CQL as CQLAlgorithm
from axiomrl.algorithms.cql import cql_loss
from axiomrl.algorithms.crossq import CrossQ as CrossQAlgorithm
from axiomrl.algorithms.crossq import crossq_loss
from axiomrl.algorithms.crr import CRR as CRRAlgorithm
from axiomrl.algorithms.crr import crr_loss
from axiomrl.algorithms.curl import CURL as CURLAlgorithm
from axiomrl.algorithms.curl import curl_loss
from axiomrl.algorithms.d4pg import D4PG as D4PGAlgorithm
from axiomrl.algorithms.d4pg import d4pg_loss
from axiomrl.algorithms.ddpg import DDPG as DDPGAlgorithm
from axiomrl.algorithms.ddpg import ddpg_loss
from axiomrl.algorithms.decision_transformer import DecisionTransformer as DecisionTransformerAlgorithm
from axiomrl.algorithms.decision_transformer import decision_transformer_loss
from axiomrl.algorithms.discrete_sac import DiscreteSAC as DiscreteSACAlgorithm
from axiomrl.algorithms.discrete_sac import discrete_sac_loss
from axiomrl.algorithms.dqn import CQLDQN as CQLDQNAlgorithm
from axiomrl.algorithms.dqn import DQN as DQNAlgorithm
from axiomrl.algorithms.dqn import AdvantageLearningDQN as AdvantageLearningDQNAlgorithm
from axiomrl.algorithms.dqn import BoltzmannDoubleDQN as BoltzmannDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import BoltzmannDQN as BoltzmannDQNAlgorithm
from axiomrl.algorithms.dqn import ClippedDoubleDQN as ClippedDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import CQLDoubleDQN as CQLDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import DoubleDQN as DoubleDQNAlgorithm
from axiomrl.algorithms.dqn import DuelingDQN as DuelingDQNAlgorithm
from axiomrl.algorithms.dqn import ExpectedDoubleDQN as ExpectedDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import ExpectedSARSA as ExpectedSARSAAlgorithm
from axiomrl.algorithms.dqn import HystereticDQN as HystereticDQNAlgorithm
from axiomrl.algorithms.dqn import MellowmaxDQN as MellowmaxDQNAlgorithm
from axiomrl.algorithms.dqn import MunchausenDoubleDQN as MunchausenDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import MunchausenDQN as MunchausenDQNAlgorithm
from axiomrl.algorithms.dqn import NoisyDQN as NoisyDQNAlgorithm
from axiomrl.algorithms.dqn import PersistentAdvantageLearningDQN as PersistentAdvantageLearningDQNAlgorithm
from axiomrl.algorithms.dqn import PrioritizedDQN as PrioritizedDQNAlgorithm
from axiomrl.algorithms.dqn import RainbowDQN as RainbowDQNAlgorithm
from axiomrl.algorithms.dqn import SoftDoubleDQN as SoftDoubleDQNAlgorithm
from axiomrl.algorithms.dqn import SoftDQN as SoftDQNAlgorithm
from axiomrl.algorithms.dqn import dqn_loss
from axiomrl.algorithms.drq import DrQ as DrQAlgorithm
from axiomrl.algorithms.drq import drq_loss
from axiomrl.algorithms.drqn import DRQN as DRQNAlgorithm
from axiomrl.algorithms.drqn import drqn_loss
from axiomrl.algorithms.drqv2 import DrQv2 as DrQv2Algorithm
from axiomrl.algorithms.drqv2 import drqv2_loss
from axiomrl.algorithms.edac import EDAC as EDACAlgorithm
from axiomrl.algorithms.edac import critic_diversity_loss, edac_loss
from axiomrl.algorithms.efficientzero import EfficientZero as EfficientZeroAlgorithm
from axiomrl.algorithms.gumbel_muzero import GumbelMuZero as GumbelMuZeroAlgorithm
from axiomrl.algorithms.her import HER as HERAlgorithm
from axiomrl.algorithms.her import her_loss
from axiomrl.algorithms.impala import IMPALA as IMPALAAlgorithm
from axiomrl.algorithms.impala import impala_loss
from axiomrl.algorithms.iql import IQL as IQLAlgorithm
from axiomrl.algorithms.iql import iql_loss
from axiomrl.algorithms.iqn import IQN as IQNAlgorithm
from axiomrl.algorithms.iqn import iqn_loss
from axiomrl.algorithms.marwil import MARWIL as MARWILAlgorithm
from axiomrl.algorithms.marwil import marwil_loss
from axiomrl.algorithms.mopo import MOPO as MOPOAlgorithm
from axiomrl.algorithms.mopo import mopo_model_loss
from axiomrl.algorithms.naf import NAF as NAFAlgorithm
from axiomrl.algorithms.naf import naf_loss
from axiomrl.algorithms.openai_es import OpenAIES as OpenAIESAlgorithm
from axiomrl.algorithms.openai_es import openai_es_loss
from axiomrl.algorithms.pets import PETS as PETSAlgorithm
from axiomrl.algorithms.pets import pets_loss
from axiomrl.algorithms.ppg import PPG as PPGAlgorithm
from axiomrl.algorithms.ppg import ppg_auxiliary_loss, ppg_loss
from axiomrl.algorithms.ppo import PPO as PPOAlgorithm
from axiomrl.algorithms.ppo import ppo_loss
from axiomrl.algorithms.qr_dqn import QRDQN as QRDQNAlgorithm
from axiomrl.algorithms.qr_dqn import qr_loss
from axiomrl.algorithms.r2d2 import R2D2 as R2D2Algorithm
from axiomrl.algorithms.r2d2 import r2d2_loss
from axiomrl.algorithms.rebrac import ReBRAC as ReBRACAlgorithm
from axiomrl.algorithms.rebrac import rebrac_loss
from axiomrl.algorithms.redq import REDQ as REDQAlgorithm
from axiomrl.algorithms.redq import redq_loss
from axiomrl.algorithms.rlpd import RLPD as RLPDAlgorithm
from axiomrl.algorithms.rlpd import rlpd_loss
from axiomrl.algorithms.sac import SAC as SACAlgorithm
from axiomrl.algorithms.sac import sac_loss
from axiomrl.algorithms.spr import SPR as SPRAlgorithm
from axiomrl.algorithms.td3 import TD3 as TD3Algorithm
from axiomrl.algorithms.td3 import td3_loss
from axiomrl.algorithms.td3_bc import TD3BC as TD3BCAlgorithm
from axiomrl.algorithms.td3_bc import td3_bc_loss
from axiomrl.algorithms.tqc import TQC as TQCAlgorithm
from axiomrl.algorithms.tqc import tqc_loss
from axiomrl.algorithms.trpo import TRPO as TRPOAlgorithm
from axiomrl.algorithms.trpo import trpo_loss
from axiomrl.algorithms.xql import XQL as XQLAlgorithm
from axiomrl.algorithms.xql import gumbel_rescale_loss, xql_loss, xql_value_loss

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
        from axiomrl.api import (
            A2C,
            APPO,
            ARS,
            AWAC,
            AWR,
            BC,
            BCQ,
            BEAR,
            C51DQN,
            CQL,
            CQLDQN,
            CRR,
            CURL,
            D4PG,
            DDPG,
            DQN,
            DRQN,
            EDAC,
            HER,
            IMPALA,
            IQL,
            IQN,
            MARWIL,
            MOPO,
            NAF,
            PETS,
            PPG,
            PPO,
            QRDQN,
            R2D2,
            REDQ,
            RLPD,
            SAC,
            SPR,
            TD3,
            TD3BC,
            TQC,
            TRPO,
            XQL,
            AdvantageLearningDQN,
            BoltzmannDoubleDQN,
            BoltzmannDQN,
            CalQL,
            ClippedDoubleDQN,
            CQLDoubleDQN,
            CrossQ,
            DecisionTransformer,
            DiscreteSAC,
            DoubleDQN,
            DrQ,
            DrQv2,
            DuelingDQN,
            EfficientZero,
            ExpectedDoubleDQN,
            ExpectedSARSA,
            GumbelMuZero,
            HystereticDQN,
            MellowmaxDQN,
            MunchausenDoubleDQN,
            MunchausenDQN,
            NoisyDQN,
            NStepDQN,
            OpenAIES,
            PersistentAdvantageLearningDQN,
            PrioritizedDQN,
            RainbowDQN,
            ReBRAC,
            SoftDoubleDQN,
            SoftDQN,
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
