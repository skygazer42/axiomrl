import pytest

from rl_training import C51DQN as RootC51DQN
from rl_training import DDPG as RootDDPG
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
from rl_training import SAC as RootSAC
from rl_training import TD3BC as RootTD3BC
from rl_training import TQC as RootTQC
from rl_training import TrainConfig
from rl_training.algorithms import (
    C51DQN,
    C51DQNAlgorithm,
    DDPG,
    DDPGAlgorithm,
    DQN,
    DQNAlgorithm,
    DoubleDQN,
    DoubleDQNAlgorithm,
    DuelingDQN,
    DuelingDQNAlgorithm,
    IQL,
    IQLAlgorithm,
    IQN,
    IQNAlgorithm,
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
    SAC,
    SACAlgorithm,
    TD3BC,
    TD3BCAlgorithm,
    TQC,
    TQCAlgorithm,
)
from rl_training.data import NStepAccumulator, RunningMeanStd, TransitionDataset
from rl_training.experiment import JsonlLogger, RunLogger


def test_package_exports_high_level_algorithms_and_config() -> None:
    assert RootC51DQN is C51DQN
    assert RootPPO is PPO
    assert RootDDPG is DDPG
    assert RootDQN is DQN
    assert RootDoubleDQN is DoubleDQN
    assert RootDuelingDQN is DuelingDQN
    assert RootIQL is IQL
    assert RootIQN is IQN
    assert RootNoisyDQN is NoisyDQN
    assert RootPrioritizedDQN is PrioritizedDQN
    assert RootRainbowDQN is RainbowDQN
    assert RootNStepDQN is NStepDQN
    assert RootQRDQN is QRDQN
    assert RootSAC is SAC
    assert RootTQC is TQC
    assert RootREDQ is REDQ
    assert RootTD3BC is TD3BC
    assert TrainConfig.__name__ == "TrainConfig"


def test_algorithms_package_exposes_high_level_and_low_level_names() -> None:
    assert C51DQN.__name__ == "C51DQN"
    assert PPO.__name__ == "PPO"
    assert DDPG.__name__ == "DDPG"
    assert DQN.__name__ == "DQN"
    assert DoubleDQN.__name__ == "DoubleDQN"
    assert DuelingDQN.__name__ == "DuelingDQN"
    assert IQL.__name__ == "IQL"
    assert IQN.__name__ == "IQN"
    assert NoisyDQN.__name__ == "NoisyDQN"
    assert PrioritizedDQN.__name__ == "PrioritizedDQN"
    assert RainbowDQN.__name__ == "RainbowDQN"
    assert NStepDQN.__name__ == "NStepDQN"
    assert QRDQN.__name__ == "QRDQN"
    assert SAC.__name__ == "SAC"
    assert TD3BC.__name__ == "TD3BC"
    assert TQC.__name__ == "TQC"
    assert REDQ.__name__ == "REDQ"
    assert C51DQNAlgorithm.__name__ == "C51DQN"
    assert PPOAlgorithm.__name__ == "PPO"
    assert DDPGAlgorithm.__name__ == "DDPG"
    assert DQNAlgorithm.__name__ == "DQN"
    assert DoubleDQNAlgorithm.__name__ == "DoubleDQN"
    assert DuelingDQNAlgorithm.__name__ == "DuelingDQN"
    assert IQLAlgorithm.__name__ == "IQL"
    assert IQNAlgorithm.__name__ == "IQN"
    assert NoisyDQNAlgorithm.__name__ == "NoisyDQN"
    assert PrioritizedDQNAlgorithm.__name__ == "PrioritizedDQN"
    assert RainbowDQNAlgorithm.__name__ == "RainbowDQN"
    assert QRDQNAlgorithm.__name__ == "QRDQN"
    assert SACAlgorithm.__name__ == "SAC"
    assert TD3BCAlgorithm.__name__ == "TD3BC"
    assert TQCAlgorithm.__name__ == "TQC"
    assert REDQAlgorithm.__name__ == "REDQ"


def test_data_package_exposes_processing_utilities() -> None:
    assert RunningMeanStd.__name__ == "RunningMeanStd"
    assert TransitionDataset.__name__ == "TransitionDataset"
    assert NStepAccumulator.__name__ == "NStepAccumulator"


def test_experiment_package_exposes_clear_logger_name_with_compat_alias(tmp_path) -> None:
    assert RunLogger.__name__ == "RunLogger"
    with pytest.deprecated_call(match="JsonlLogger is deprecated; use RunLogger instead"):
        logger = JsonlLogger(tmp_path)
    try:
        assert isinstance(logger, RunLogger)
    finally:
        logger.close()
