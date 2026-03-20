from rl_training.experiment import registry_core


def test_registry_core_uses_split_type_and_support_modules() -> None:
    assert registry_core.AlgorithmSpec.__module__ == "rl_training.experiment.registry_types"
    assert registry_core._prepare_observation.__module__ == "rl_training.experiment.registry_support"
    assert registry_core._format_action_output.__module__ == "rl_training.experiment.registry_support"
