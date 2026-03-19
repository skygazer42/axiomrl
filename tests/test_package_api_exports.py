import importlib

import pytest

import rl_training as root_package
from rl_training.api import A2C as ApiA2C
from rl_training.api import ARS as ApiARS
from rl_training.api import BC as ApiBC
from rl_training.api import CQL as ApiCQL
from rl_training.api import DQN as ApiDQN
from rl_training.api import DiscreteSAC as ApiDiscreteSAC
from rl_training.api import IQL as ApiIQL
from rl_training.api import PPO as ApiPPO
from rl_training.api import SAC as ApiSAC
from rl_training.api import TD3 as ApiTD3
from rl_training.api import TRPO as ApiTRPO
from rl_training.contrib import RecurrentPPO, RecurrentPPOAlgorithm
from rl_training.experiment import JsonlLogger, RunLogger
from rl_training.experiment.config import TrainConfig


def test_root_package_exports_only_stable_core_algorithms_by_default() -> None:
    assert root_package.A2C is ApiA2C
    assert root_package.BC is ApiBC
    assert root_package.CQL is ApiCQL
    assert root_package.DQN is ApiDQN
    assert root_package.DiscreteSAC is ApiDiscreteSAC
    assert root_package.IQL is ApiIQL
    assert root_package.PPO is ApiPPO
    assert root_package.SAC is ApiSAC
    assert root_package.TD3 is ApiTD3
    assert root_package.TRPO is ApiTRPO
    assert root_package.TrainConfig is TrainConfig


def test_root_package_exposes_stability_namespaces() -> None:
    core_module = importlib.import_module("rl_training.core")
    experimental_module = importlib.import_module("rl_training.experimental")

    assert root_package.core is core_module
    assert root_package.experimental is experimental_module
    assert root_package.contrib.RecurrentPPO is RecurrentPPO
    assert root_package.contrib.RecurrentPPOAlgorithm is RecurrentPPOAlgorithm


def test_core_module_lists_stable_algorithm_exports() -> None:
    core_module = importlib.import_module("rl_training.core")

    assert core_module.STABLE_ALGORITHMS == (
        "A2C",
        "BC",
        "CQL",
        "DQN",
        "DiscreteSAC",
        "IQL",
        "PPO",
        "SAC",
        "TD3",
        "TRPO",
    )
    assert core_module.PPO is ApiPPO
    assert core_module.SAC is ApiSAC
    assert core_module.TD3 is ApiTD3


def test_experimental_module_reexports_advanced_algorithms() -> None:
    experimental_module = importlib.import_module("rl_training.experimental")

    assert experimental_module.ARS is ApiARS
    assert "ARS" in experimental_module.EXPERIMENTAL_ALGORITHMS
    assert "PPO" not in experimental_module.EXPERIMENTAL_ALGORITHMS


def test_legacy_root_experimental_imports_warn_and_resolve() -> None:
    with pytest.deprecated_call(match="stable root API"):
        assert getattr(root_package, "ARS") is ApiARS


def test_root_package_all_matches_stable_surface() -> None:
    assert root_package.__all__ == [
        "__version__",
        "STABLE_ALGORITHMS",
        "A2C",
        "BC",
        "CQL",
        "DQN",
        "DiscreteSAC",
        "IQL",
        "PPO",
        "SAC",
        "TD3",
        "TRPO",
        "TrainConfig",
        "core",
        "experimental",
        "contrib",
    ]


def test_experiment_package_exposes_clear_logger_name_with_compat_alias(tmp_path) -> None:
    assert RunLogger.__name__ == "RunLogger"
    with pytest.deprecated_call(match="JsonlLogger is deprecated; use RunLogger instead"):
        logger = JsonlLogger(tmp_path)
    try:
        assert isinstance(logger, RunLogger)
    finally:
        logger.close()
