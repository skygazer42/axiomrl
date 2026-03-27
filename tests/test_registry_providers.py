from rl_training.experiment.registry import get_algorithm_spec, list_algorithm_specs
from rl_training.experiment.registry_core import AlgorithmSpec
from rl_training.experiment.registry_providers import (
    ACTOR_CRITIC_SPECS,
    ALL_SPEC_GROUPS,
    CONTRIB_SPECS,
    GOAL_CONDITIONED_SPECS,
    OFFLINE_SPECS,
    ON_POLICY_SPECS,
    VALUE_BASED_SPECS,
    WORLD_MODEL_SPECS,
)


def test_registry_provider_groups_cover_public_registry_without_duplicates() -> None:
    grouped_specs: dict[str, AlgorithmSpec] = {}

    for group in ALL_SPEC_GROUPS.values():
        duplicate_names = set(grouped_specs).intersection(group)
        assert duplicate_names == set()
        grouped_specs.update(group)

    assert set(grouped_specs) == {spec.name for spec in list_algorithm_specs()}


def test_registry_provider_groups_expose_expected_algorithm_families() -> None:
    assert ON_POLICY_SPECS["ppo"] is get_algorithm_spec("ppo")
    assert VALUE_BASED_SPECS["dqn"] is get_algorithm_spec("dqn")
    assert ACTOR_CRITIC_SPECS["sac"] is get_algorithm_spec("sac")
    assert OFFLINE_SPECS["iql"] is get_algorithm_spec("iql")
    assert WORLD_MODEL_SPECS["dreamer"] is get_algorithm_spec("dreamer")
    assert GOAL_CONDITIONED_SPECS["her"] is get_algorithm_spec("her")
    assert CONTRIB_SPECS["recurrent_ppo"] is get_algorithm_spec("recurrent_ppo")
