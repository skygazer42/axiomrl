import pytest

from rl_training import AWAC as RootAWAC
from rl_training import AWR as RootAWR
from rl_training import MARWIL as RootMARWIL
from rl_training import HER as RootHER
from rl_training.api import contrib as api_contrib
from rl_training import BEAR as RootBEAR
from rl_training import BC as RootBC
from rl_training import BCQ as RootBCQ
from rl_training import C51DQN as RootC51DQN
from rl_training import CalQL as RootCalQL
from rl_training import CrossQ as RootCrossQ
from rl_training import CRR as RootCRR
from rl_training import DDPG as RootDDPG
from rl_training import EDAC as RootEDAC
from rl_training import DrQv2 as RootDrQv2
from rl_training import DiscreteSAC as RootDiscreteSAC
from rl_training import DQN as RootDQN
from rl_training import DoubleDQN as RootDoubleDQN
from rl_training import DuelingDQN as RootDuelingDQN
from rl_training import IQL as RootIQL
from rl_training import IQN as RootIQN
from rl_training import NStepDQN as RootNStepDQN
from rl_training import NoisyDQN as RootNoisyDQN
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
    BCQ,
    BCQAlgorithm,
    C51DQN,
    C51DQNAlgorithm,
    CalQL,
    CalQLAlgorithm,
    CrossQ,
    CrossQAlgorithm,
    CRR,
    CRRAlgorithm,
    CQL,
    CQLAlgorithm,
    DDPG,
    DDPGAlgorithm,
    EDAC,
    EDACAlgorithm,
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
    XQL,
    XQLAlgorithm,
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
from rl_training.data import NStepAccumulator, RunningMeanStd, TransitionDataset, compute_discounted_returns_to_go
from rl_training.experiment import JsonlLogger, RunLogger


def test_package_exports_high_level_algorithms_and_config() -> None:
    assert RootAWR is AWR
    assert RootAWAC is AWAC
    assert RootMARWIL is MARWIL
    assert RootBEAR is BEAR
    assert RootBC is BC
    assert RootBCQ is BCQ
    assert RootC51DQN is C51DQN
    assert RootCalQL is CalQL
    assert RootCrossQ is CrossQ
    assert RootCRR is CRR
    assert RootPPO is PPO
    assert RootDDPG is DDPG
    assert RootEDAC is EDAC
    assert RootDrQv2 is DrQv2
    assert RootDiscreteSAC is DiscreteSAC
    assert RootDQN is DQN
    assert RootDoubleDQN is DoubleDQN
    assert RootDuelingDQN is DuelingDQN
    assert RootHER is HER
    assert RootIQL is IQL
    assert RootIQN is IQN
    assert RootXQL is XQL
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
    assert AWR.__name__ == "AWR"
    assert AWAC.__name__ == "AWAC"
    assert MARWIL.__name__ == "MARWIL"
    assert BEAR.__name__ == "BEAR"
    assert BC.__name__ == "BC"
    assert BCQ.__name__ == "BCQ"
    assert C51DQN.__name__ == "C51DQN"
    assert CalQL.__name__ == "CalQL"
    assert CrossQ.__name__ == "CrossQ"
    assert CRR.__name__ == "CRR"
    assert PPO.__name__ == "PPO"
    assert DDPG.__name__ == "DDPG"
    assert EDAC.__name__ == "EDAC"
    assert DrQv2.__name__ == "DrQv2"
    assert DQN.__name__ == "DQN"
    assert DoubleDQN.__name__ == "DoubleDQN"
    assert DuelingDQN.__name__ == "DuelingDQN"
    assert HER.__name__ == "HER"
    assert IQL.__name__ == "IQL"
    assert IQN.__name__ == "IQN"
    assert XQL.__name__ == "XQL"
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
    assert AWRAlgorithm.__name__ == "AWR"
    assert AWACAlgorithm.__name__ == "AWAC"
    assert MARWILAlgorithm.__name__ == "MARWIL"
    assert BEARAlgorithm.__name__ == "BEAR"
    assert BCAlgorithm.__name__ == "BC"
    assert BCQAlgorithm.__name__ == "BCQ"
    assert C51DQNAlgorithm.__name__ == "C51DQN"
    assert CalQLAlgorithm.__name__ == "CalQL"
    assert CrossQAlgorithm.__name__ == "CrossQ"
    assert CRRAlgorithm.__name__ == "CRR"
    assert PPOAlgorithm.__name__ == "PPO"
    assert DDPGAlgorithm.__name__ == "DDPG"
    assert EDACAlgorithm.__name__ == "EDAC"
    assert DrQv2Algorithm.__name__ == "DrQv2"
    assert DQNAlgorithm.__name__ == "DQN"
    assert DoubleDQNAlgorithm.__name__ == "DoubleDQN"
    assert DuelingDQNAlgorithm.__name__ == "DuelingDQN"
    assert HERAlgorithm.__name__ == "HER"
    assert IQLAlgorithm.__name__ == "IQL"
    assert IQNAlgorithm.__name__ == "IQN"
    assert XQLAlgorithm.__name__ == "XQL"
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
