import pytest

from rl_training import AWAC as RootAWAC
from rl_training import AWR as RootAWR
from rl_training import ARS as RootARS
from rl_training import OpenAIES as RootOpenAIES
from rl_training import MARWIL as RootMARWIL
from rl_training import HER as RootHER
from rl_training.api import contrib as api_contrib
from rl_training import BEAR as RootBEAR
from rl_training import BC as RootBC
from rl_training import DecisionTransformer as RootDecisionTransformer
from rl_training import BCQ as RootBCQ
from rl_training import C51DQN as RootC51DQN
from rl_training import CalQL as RootCalQL
from rl_training import CURL as RootCURL
from rl_training import CrossQ as RootCrossQ
from rl_training import CRR as RootCRR
from rl_training import D4PG as RootD4PG
from rl_training import DDPG as RootDDPG
from rl_training import DRQN as RootDRQN
from rl_training import R2D2 as RootR2D2
from rl_training import EDAC as RootEDAC
from rl_training import DrQ as RootDrQ
from rl_training import DrQv2 as RootDrQv2
from rl_training import DiscreteSAC as RootDiscreteSAC
from rl_training import DQN as RootDQN
from rl_training import DoubleDQN as RootDoubleDQN
from rl_training import DuelingDQN as RootDuelingDQN
from rl_training import IQL as RootIQL
from rl_training import IQN as RootIQN
from rl_training import IMPALA as RootIMPALA
from rl_training import APPO as RootAPPO
from rl_training import MOPO as RootMOPO
from rl_training import PETS as RootPETS
from rl_training import NAF as RootNAF
from rl_training import NStepDQN as RootNStepDQN
from rl_training import NoisyDQN as RootNoisyDQN
from rl_training import PPG as RootPPG
from rl_training import PrioritizedDQN as RootPrioritizedDQN
from rl_training import PPO as RootPPO
from rl_training import QRDQN as RootQRDQN
from rl_training import RainbowDQN as RootRainbowDQN
from rl_training import REDQ as RootREDQ
from rl_training import RLPD as RootRLPD
from rl_training import ReBRAC as RootReBRAC
from rl_training import SAC as RootSAC
from rl_training import CQL as RootCQL
from rl_training import XQL as RootXQL
from rl_training import TD3BC as RootTD3BC
from rl_training import TRPO as RootTRPO
from rl_training import TQC as RootTQC
from rl_training import TrainConfig
from rl_training import contrib as root_contrib
from rl_training.algorithms import (
    ARS,
    ARSAlgorithm,
    OpenAIES,
    OpenAIESAlgorithm,
    AWR,
    AWRAlgorithm,
    AWAC,
    AWACAlgorithm,
    MARWIL,
    MARWILAlgorithm,
    BEAR,
    BEARAlgorithm,
    BC,
    BCAlgorithm,
    DecisionTransformer,
    DecisionTransformerAlgorithm,
    BCQ,
    BCQAlgorithm,
    C51DQN,
    C51DQNAlgorithm,
    CalQL,
    CalQLAlgorithm,
    CURL,
    CURLAlgorithm,
    CrossQ,
    CrossQAlgorithm,
    CRR,
    CRRAlgorithm,
    D4PG,
    D4PGAlgorithm,
    CQL,
    CQLAlgorithm,
    DDPG,
    DDPGAlgorithm,
    DRQN,
    DRQNAlgorithm,
    R2D2,
    R2D2Algorithm,
    EDAC,
    EDACAlgorithm,
    DrQ,
    DrQAlgorithm,
    DrQv2,
    DrQv2Algorithm,
    DiscreteSAC,
    DiscreteSACAlgorithm,
    DQN,
    DQNAlgorithm,
    DoubleDQN,
    DoubleDQNAlgorithm,
    DuelingDQN,
    DuelingDQNAlgorithm,
    HER,
    HERAlgorithm,
    IQL,
    IQLAlgorithm,
    IQN,
    IQNAlgorithm,
    IMPALA,
    IMPALAAlgorithm,
    APPO,
    APPOAlgorithm,
    MOPO,
    MOPOAlgorithm,
    PETS,
    PETSAlgorithm,
    NAF,
    NAFAlgorithm,
    XQL,
    XQLAlgorithm,
    PPG,
    PPGAlgorithm,
    NStepDQN,
    NoisyDQN,
    NoisyDQNAlgorithm,
    PPO,
    PPOAlgorithm,
    PrioritizedDQN,
    PrioritizedDQNAlgorithm,
    QRDQN,
    QRDQNAlgorithm,
    RainbowDQN,
    RainbowDQNAlgorithm,
    REDQ,
    REDQAlgorithm,
    RLPD,
    RLPDAlgorithm,
    ReBRAC,
    ReBRACAlgorithm,
    SAC,
    SACAlgorithm,
    TD3BC,
    TD3BCAlgorithm,
    TRPO,
    TRPOAlgorithm,
    TQC,
    TQCAlgorithm,
)
from rl_training.contrib import RecurrentPPO, RecurrentPPOAlgorithm
from rl_training.data import (
    NStepAccumulator,
    RecurrentReplayBuffer,
    PrioritizedRecurrentReplayBuffer,
    RunningMeanStd,
    TransitionDataset,
    compute_discounted_returns_to_go,
)
from rl_training.experiment import JsonlLogger, RunLogger


def test_package_exports_high_level_algorithms_and_config() -> None:
    assert RootARS is ARS
    assert RootOpenAIES is OpenAIES
    assert RootAWR is AWR
    assert RootAWAC is AWAC
    assert RootMARWIL is MARWIL
    assert RootBEAR is BEAR
    assert RootBC is BC
    assert RootDecisionTransformer is DecisionTransformer
    assert RootBCQ is BCQ
    assert RootC51DQN is C51DQN
    assert RootCalQL is CalQL
    assert RootCURL is CURL
    assert RootCrossQ is CrossQ
    assert RootCRR is CRR
    assert RootD4PG is D4PG
    assert RootPPO is PPO
    assert RootDDPG is DDPG
    assert RootDRQN is DRQN
    assert RootR2D2 is R2D2
    assert RootEDAC is EDAC
    assert RootDrQ is DrQ
    assert RootDrQv2 is DrQv2
    assert RootDiscreteSAC is DiscreteSAC
    assert RootDQN is DQN
    assert RootDoubleDQN is DoubleDQN
    assert RootDuelingDQN is DuelingDQN
    assert RootHER is HER
    assert RootIQL is IQL
    assert RootIQN is IQN
    assert RootIMPALA is IMPALA
    assert RootAPPO is APPO
    assert RootMOPO is MOPO
    assert RootPETS is PETS
    assert RootNAF is NAF
    assert RootXQL is XQL
    assert RootPPG is PPG
    assert RootNoisyDQN is NoisyDQN
    assert RootPrioritizedDQN is PrioritizedDQN
    assert RootRainbowDQN is RainbowDQN
    assert RootNStepDQN is NStepDQN
    assert RootQRDQN is QRDQN
    assert RootSAC is SAC
    assert RootCQL is CQL
    assert RootTQC is TQC
    assert RootREDQ is REDQ
    assert RootRLPD is RLPD
    assert RootReBRAC is ReBRAC
    assert RootTD3BC is TD3BC
    assert RootTRPO is TRPO
    assert TrainConfig.__name__ == "TrainConfig"


def test_algorithms_package_exposes_high_level_and_low_level_names() -> None:
    assert ARS.__name__ == "ARS"
    assert OpenAIES.__name__ == "OpenAIES"
    assert AWR.__name__ == "AWR"
    assert AWAC.__name__ == "AWAC"
    assert MARWIL.__name__ == "MARWIL"
    assert BEAR.__name__ == "BEAR"
    assert BC.__name__ == "BC"
    assert DecisionTransformer.__name__ == "DecisionTransformer"
    assert BCQ.__name__ == "BCQ"
    assert C51DQN.__name__ == "C51DQN"
    assert CalQL.__name__ == "CalQL"
    assert CURL.__name__ == "CURL"
    assert CrossQ.__name__ == "CrossQ"
    assert CRR.__name__ == "CRR"
    assert D4PG.__name__ == "D4PG"
    assert PPO.__name__ == "PPO"
    assert DDPG.__name__ == "DDPG"
    assert DRQN.__name__ == "DRQN"
    assert R2D2.__name__ == "R2D2"
    assert EDAC.__name__ == "EDAC"
    assert DrQ.__name__ == "DrQ"
    assert DrQv2.__name__ == "DrQv2"
    assert DQN.__name__ == "DQN"
    assert DoubleDQN.__name__ == "DoubleDQN"
    assert DuelingDQN.__name__ == "DuelingDQN"
    assert HER.__name__ == "HER"
    assert IQL.__name__ == "IQL"
    assert IQN.__name__ == "IQN"
    assert IMPALA.__name__ == "IMPALA"
    assert APPO.__name__ == "APPO"
    assert MOPO.__name__ == "MOPO"
    assert PETS.__name__ == "PETS"
    assert NAF.__name__ == "NAF"
    assert XQL.__name__ == "XQL"
    assert PPG.__name__ == "PPG"
    assert NoisyDQN.__name__ == "NoisyDQN"
    assert PrioritizedDQN.__name__ == "PrioritizedDQN"
    assert RainbowDQN.__name__ == "RainbowDQN"
    assert NStepDQN.__name__ == "NStepDQN"
    assert QRDQN.__name__ == "QRDQN"
    assert SAC.__name__ == "SAC"
    assert CQL.__name__ == "CQL"
    assert DiscreteSAC.__name__ == "DiscreteSAC"
    assert ReBRAC.__name__ == "ReBRAC"
    assert RLPD.__name__ == "RLPD"
    assert TRPO.__name__ == "TRPO"
    assert TD3BC.__name__ == "TD3BC"
    assert TQC.__name__ == "TQC"
    assert REDQ.__name__ == "REDQ"
    assert ARSAlgorithm.__name__ == "ARS"
    assert OpenAIESAlgorithm.__name__ == "OpenAIES"
    assert AWRAlgorithm.__name__ == "AWR"
    assert AWACAlgorithm.__name__ == "AWAC"
    assert MARWILAlgorithm.__name__ == "MARWIL"
    assert BEARAlgorithm.__name__ == "BEAR"
    assert BCAlgorithm.__name__ == "BC"
    assert DecisionTransformerAlgorithm.__name__ == "DecisionTransformer"
    assert BCQAlgorithm.__name__ == "BCQ"
    assert C51DQNAlgorithm.__name__ == "C51DQN"
    assert CalQLAlgorithm.__name__ == "CalQL"
    assert CURLAlgorithm.__name__ == "CURL"
    assert CrossQAlgorithm.__name__ == "CrossQ"
    assert CRRAlgorithm.__name__ == "CRR"
    assert D4PGAlgorithm.__name__ == "D4PG"
    assert PPOAlgorithm.__name__ == "PPO"
    assert DDPGAlgorithm.__name__ == "DDPG"
    assert DRQNAlgorithm.__name__ == "DRQN"
    assert R2D2Algorithm.__name__ == "R2D2"
    assert EDACAlgorithm.__name__ == "EDAC"
    assert DrQAlgorithm.__name__ == "DrQ"
    assert DrQv2Algorithm.__name__ == "DrQv2"
    assert DQNAlgorithm.__name__ == "DQN"
    assert DoubleDQNAlgorithm.__name__ == "DoubleDQN"
    assert DuelingDQNAlgorithm.__name__ == "DuelingDQN"
    assert HERAlgorithm.__name__ == "HER"
    assert IQLAlgorithm.__name__ == "IQL"
    assert IQNAlgorithm.__name__ == "IQN"
    assert IMPALAAlgorithm.__name__ == "IMPALA"
    assert APPOAlgorithm.__name__ == "APPO"
    assert MOPOAlgorithm.__name__ == "MOPO"
    assert NAFAlgorithm.__name__ == "NAF"
    assert XQLAlgorithm.__name__ == "XQL"
    assert PPGAlgorithm.__name__ == "PPG"
    assert NoisyDQNAlgorithm.__name__ == "NoisyDQN"
    assert PrioritizedDQNAlgorithm.__name__ == "PrioritizedDQN"
    assert RainbowDQNAlgorithm.__name__ == "RainbowDQN"
    assert QRDQNAlgorithm.__name__ == "QRDQN"
    assert SACAlgorithm.__name__ == "SAC"
    assert CQLAlgorithm.__name__ == "CQL"
    assert DiscreteSACAlgorithm.__name__ == "DiscreteSAC"
    assert RLPDAlgorithm.__name__ == "RLPD"
    assert ReBRACAlgorithm.__name__ == "ReBRAC"
    assert TRPOAlgorithm.__name__ == "TRPO"
    assert TD3BCAlgorithm.__name__ == "TD3BC"
    assert TQCAlgorithm.__name__ == "TQC"
    assert REDQAlgorithm.__name__ == "REDQ"


def test_data_package_exposes_processing_utilities() -> None:
    assert compute_discounted_returns_to_go.__name__ == "compute_discounted_returns_to_go"
    assert RunningMeanStd.__name__ == "RunningMeanStd"
    assert TransitionDataset.__name__ == "TransitionDataset"
    assert NStepAccumulator.__name__ == "NStepAccumulator"
    assert RecurrentReplayBuffer.__name__ == "RecurrentReplayBuffer"
    assert PrioritizedRecurrentReplayBuffer.__name__ == "PrioritizedRecurrentReplayBuffer"


def test_root_package_exposes_contrib_namespace() -> None:
    assert root_contrib.RecurrentPPO is RecurrentPPO
    assert root_contrib.RecurrentPPOAlgorithm is RecurrentPPOAlgorithm


def test_api_package_exposes_contrib_namespace() -> None:
    assert api_contrib.RecurrentPPO is RecurrentPPO
    assert api_contrib.RecurrentPPOAlgorithm is RecurrentPPOAlgorithm


def test_experiment_package_exposes_clear_logger_name_with_compat_alias(tmp_path) -> None:
    assert RunLogger.__name__ == "RunLogger"
    with pytest.deprecated_call(match="JsonlLogger is deprecated; use RunLogger instead"):
        logger = JsonlLogger(tmp_path)
    try:
        assert isinstance(logger, RunLogger)
    finally:
        logger.close()
