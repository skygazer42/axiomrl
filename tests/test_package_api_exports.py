import pytest

from rl_training import DQN as RootDQN
from rl_training import PPO as RootPPO
from rl_training import SAC as RootSAC
from rl_training import TrainConfig
from rl_training.algorithms import DQN, DQNAlgorithm, PPO, PPOAlgorithm, SAC, SACAlgorithm
from rl_training.experiment import JsonlLogger, RunLogger


def test_package_exports_high_level_algorithms_and_config() -> None:
    assert RootPPO is PPO
    assert RootDQN is DQN
    assert RootSAC is SAC
    assert TrainConfig.__name__ == "TrainConfig"


def test_algorithms_package_exposes_high_level_and_low_level_names() -> None:
    assert PPO.__name__ == "PPO"
    assert DQN.__name__ == "DQN"
    assert SAC.__name__ == "SAC"
    assert PPOAlgorithm.__name__ == "PPO"
    assert DQNAlgorithm.__name__ == "DQN"
    assert SACAlgorithm.__name__ == "SAC"


def test_experiment_package_exposes_clear_logger_name_with_compat_alias(tmp_path) -> None:
    assert RunLogger.__name__ == "RunLogger"
    with pytest.deprecated_call(match="JsonlLogger is deprecated; use RunLogger instead"):
        logger = JsonlLogger(tmp_path)
    try:
        assert isinstance(logger, RunLogger)
    finally:
        logger.close()
