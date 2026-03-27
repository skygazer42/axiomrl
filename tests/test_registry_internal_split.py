from rl_training.experiment import registry_core
from rl_training.experiment.registry_providers import actor_critic as actor_critic_provider
from rl_training.experiment.registry_providers import offline as offline_provider
from rl_training.experiment.registry_providers import on_policy as on_policy_provider
from rl_training.experiment.registry_providers import value_based as value_based_provider
from rl_training.experiment.registry_providers import world_model as world_model_provider


def test_registry_core_uses_split_type_and_support_modules() -> None:
    assert registry_core.AlgorithmSpec.__module__ == "rl_training.experiment.registry_types"
    assert registry_core._prepare_observation.__module__ == "rl_training.experiment.registry_support"
    assert registry_core._format_action_output.__module__ == "rl_training.experiment.registry_support"
    assert registry_core._evaluate_ppo.__module__ == "rl_training.experiment.registry_evaluators"
    assert registry_core._predict_ppo.__module__ == "rl_training.experiment.registry_predictors"
    assert registry_core._load_dqn_algorithm.__module__ == "rl_training.experiment.registry_dqn_loaders"
    assert registry_core._load_c51_dqn_algorithm.__module__ == "rl_training.experiment.registry_dqn_loaders"
    assert registry_core._load_fqf_algorithm.__module__ == "rl_training.experiment.registry_dqn_loaders"
    assert registry_core._load_iql_algorithm.__module__ == "rl_training.experiment.registry_continuous_loaders"
    assert registry_core._load_sac_algorithm.__module__ == "rl_training.experiment.registry_continuous_loaders"
    assert registry_core._load_drq_algorithm.__module__ == "rl_training.experiment.registry_continuous_loaders"
    assert registry_core._load_recurrent_ppo_algorithm.__module__ == "rl_training.experiment.registry_recurrent_loaders"
    assert registry_core._load_drqn_algorithm.__module__ == "rl_training.experiment.registry_recurrent_loaders"
    assert registry_core._load_agent57_algorithm.__module__ == "rl_training.experiment.registry_recurrent_loaders"
    assert registry_core._load_a2c_algorithm.__module__ == "rl_training.experiment.registry_policy_loaders"
    assert registry_core._load_ppo_algorithm.__module__ == "rl_training.experiment.registry_policy_loaders"
    assert (
        registry_core._load_decision_transformer_algorithm.__module__
        == "rl_training.experiment.registry_policy_loaders"
    )
    assert registry_core._load_mopo_algorithm.__module__ == "rl_training.experiment.registry_specialized_loaders"
    assert registry_core._load_dreamer_algorithm.__module__ == "rl_training.experiment.registry_specialized_loaders"
    assert registry_core._load_muzero_algorithm.__module__ == "rl_training.experiment.registry_specialized_loaders"
    assert (
        registry_core._load_discrete_sac_algorithm.__module__ == "rl_training.experiment.registry_specialized_loaders"
    )
    assert registry_core._load_bcq_algorithm.__module__ == "rl_training.experiment.registry_offline_loaders"
    assert registry_core._load_awac_algorithm.__module__ == "rl_training.experiment.registry_offline_loaders"
    assert registry_core._load_xql_algorithm.__module__ == "rl_training.experiment.registry_offline_loaders"
    assert registry_core._load_rebrac_algorithm.__module__ == "rl_training.experiment.registry_offline_loaders"


def test_on_policy_provider_uses_direct_spec_module() -> None:
    assert "_ALGORITHM_REGISTRY" not in vars(on_policy_provider)
    assert on_policy_provider.ON_POLICY_SPECS["ppo"].name == "ppo"


def test_offline_provider_uses_direct_spec_module() -> None:
    assert "_ALGORITHM_REGISTRY" not in vars(offline_provider)
    assert offline_provider.OFFLINE_SPECS["iql"].name == "iql"


def test_world_model_provider_uses_direct_spec_module() -> None:
    assert "_ALGORITHM_REGISTRY" not in vars(world_model_provider)
    assert world_model_provider.WORLD_MODEL_SPECS["dreamer"].name == "dreamer"


def test_actor_critic_provider_uses_direct_spec_module() -> None:
    assert "_ALGORITHM_REGISTRY" not in vars(actor_critic_provider)
    assert actor_critic_provider.ACTOR_CRITIC_SPECS["sac"].name == "sac"


def test_value_based_provider_uses_direct_spec_module() -> None:
    assert "_ALGORITHM_REGISTRY" not in vars(value_based_provider)
    assert value_based_provider.VALUE_BASED_SPECS["dqn"].name == "dqn"
